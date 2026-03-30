"""
FastAPI application for AI MCQ Generator
Includes: input validation, rate limiting, caching, compression, health checks
"""
import os
import sys
import json
import traceback
import gzip
import io
import base64, tempfile
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, HttpUrl, Field
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
import aiohttp
import aiofiles

from config import settings
from logger import logger
from cache import mcq_cache
from rate_limiter import rate_limiter
from mcq_core import extract_text, generate_balanced_mcqs, save_mcqs_txt, save_mcqs_pdf, save_mcqs_docx


# ===================================================================
#                         SETUP
# ===================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = FastAPI(title="AI MCQ Generator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
#                    INPUT VALIDATION MODELS
# ===================================================================

class MCQGenerationRequest(BaseModel):
    """Validated request model for MCQ generation"""
    url_input: str = ""
    total_questions: int = Field(..., ge=settings.min_questions, le=settings.max_questions)
    co_list: str
    topic_name: str = ""
    
    @validator('total_questions')
    def validate_question_count(cls, v):
        if not settings.min_questions <= v <= settings.max_questions:
            raise ValueError(
                f'Questions must be between {settings.min_questions} and {settings.max_questions}'
            )
        return v
    
    @validator('co_list')
    def validate_cos(cls, v):
        cos = [line.strip() for line in v.split('\n') if line.strip()]
        if not cos:
            raise ValueError('At least one Course Outcome is required')
        if len(cos) > settings.max_cos:
            raise ValueError(f'Maximum {settings.max_cos} Course Outcomes allowed')
        # Truncate each CO to max length
        return '\n'.join(co[:settings.max_co_length] for co in cos)
    
    @validator('topic_name')
    def sanitize_topic_name(cls, v):
        if not v:
            return v
        # Remove dangerous characters, keep only alphanumeric, dash, underscore
        import re
        sanitized = re.sub(r'[^\w\-]', '_', v)
        return sanitized[:settings.max_topic_name_length]
    
    @validator('url_input')
    def validate_url(cls, v):
        if not v or not v.strip():
            return ""
        # Basic URL validation
        v = v.strip()
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


# ===================================================================
#                         HELPER FUNCTIONS
# ===================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in settings.allowed_extensions


async def extract_text_from_url(url: str) -> str:
    """Extract text from URL asynchronously"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                html = await resp.text()
        
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.extract()
        
        lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
        return "\n".join(lines)
    
    except Exception as e:
        logger.error(f"Failed to extract text from URL {url}: {e}")
        raise ValueError(f"Failed to extract data from URL: {e}")


async def save_uploaded_file_streaming(file: UploadFile, filepath: str) -> None:
    """
    Save uploaded file using streaming to avoid memory issues
    Handles large files efficiently
    """
    try:
        async with aiofiles.open(filepath, 'wb') as f:
            while chunk := await file.read(settings.chunk_size_bytes):
                await f.write(chunk)
        logger.info(f"Saved uploaded file: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save file {filepath}: {e}")
        raise


def get_client_ip(request: Request) -> str:
    """Extract client IP from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def error_response(msg: str, status: int = 400) -> JSONResponse:
    """Create error response"""
    logger.warning(f"Error response: {msg} (status={status})")
    return JSONResponse({"error": msg}, status_code=status)


# ===================================================================
#                         HEALTH CHECK
# ===================================================================

@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint
    Checks API connectivity and system status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_size": mcq_cache.size(),
    }
    
    # Check Groq API connectivity
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            }
            async with session.get(
                "https://api.groq.com/openai/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    health_status["groq_api"] = "up"
                else:
                    health_status["groq_api"] = "degraded"
                    health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Groq API health check failed: {e}")
        health_status["groq_api"] = "down"
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(health_status, status_code=status_code)


# ===================================================================
#                    MAIN GENERATION ENDPOINT
# ===================================================================

@app.post("/generate")
async def generate_mcqs(
    request: Request,
    url_input: str = Form(default=""),
    total_questions: int = Form(...),
    co_list: str = Form(...),
    topic_name: str = Form(default=""),
    file: UploadFile = File(default=None),
):
    """
    Generate MCQs with validation, rate limiting, and caching
    """
    try:
        # Rate limiting
        client_ip = get_client_ip(request)
        allowed, error_msg = rate_limiter.is_allowed(client_ip)
        if not allowed:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return error_response(error_msg, 429)

        # Validate inputs
        try:
            validated = MCQGenerationRequest(
                url_input=url_input,
                total_questions=total_questions,
                co_list=co_list,
                topic_name=topic_name
            )
        except Exception as e:
            return error_response(f"Invalid input: {str(e)}", 400)

        # Extract text
        if validated.url_input:
            try:
                text = await extract_text_from_url(validated.url_input)
                base_name = validated.topic_name or "generated_from_url"
            except Exception as e:
                return error_response(f"Error fetching URL: {e}", 400)
        else:
            if not file or not file.filename:
                return error_response("No file uploaded and no URL provided", 400)

            if not allowed_file(file.filename):
                return error_response(
                    f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions).upper()}",
                    400
                )

            # File size check
            file.file.seek(0, 2)
            file_size_mb = file.file.tell() / (1024 * 1024)
            file.file.seek(0)

            if file_size_mb > settings.max_file_size_mb:
                return error_response(
                    f"File too large. Maximum size: {settings.max_file_size_mb}MB",
                    400
                )

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            try:
                await save_uploaded_file_streaming(file, filepath)
                text = extract_text(filepath)
                base_name = validated.topic_name or filename.rsplit(".", 1)[0]
            except Exception as e:
                logger.error(f"Text extraction failed: {e}")
                return error_response(f"Error extracting text: {e}", 500)
            finally:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass

        # Parse COs
        co_entries = [line.strip() for line in validated.co_list.split("\n") if line.strip()]

        # Cache check
        cached_result = mcq_cache.get(text, co_entries, validated.total_questions)
        if cached_result:
            mapped_mcqs = cached_result["mapped_questions"]
        else:
            try:
                result = await generate_balanced_mcqs(text, co_entries, validated.total_questions)
                mapped_mcqs = result.get("mapped_questions", [])
                mcq_cache.set(text, co_entries, validated.total_questions, result)
            except Exception as e:
                logger.error(f"MCQ generation failed: {e}")
                return error_response("Error generating MCQs", 500)

        # File naming
        ist = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime.now(ist).strftime("%Y%m%d_%H%M%S")
        safe_base = secure_filename(base_name)

        txt_name = f"{safe_base}_{timestamp}.txt"
        pdf_name = f"{safe_base}_{timestamp}.pdf"
        json_name = f"{safe_base}_{timestamp}.json"
        docx_name = f"{safe_base}_{timestamp}.docx"

        # Save + encode
        try:
            with tempfile.TemporaryDirectory() as tmp:
                save_mcqs_txt(mapped_mcqs, tmp, txt_name)
                save_mcqs_pdf(mapped_mcqs, tmp, pdf_name)
                save_mcqs_docx(mapped_mcqs, tmp, docx_name)

                with open(os.path.join(tmp, txt_name), 'rb') as f:
                    txt_b64 = base64.b64encode(f.read()).decode()
                with open(os.path.join(tmp, pdf_name), 'rb') as f:
                    pdf_b64 = base64.b64encode(f.read()).decode()
                with open(os.path.join(tmp, docx_name), 'rb') as f:
                    docx_b64 = base64.b64encode(f.read()).decode()

                json_b64 = base64.b64encode(
                    json.dumps(mapped_mcqs).encode()
                ).decode()

                return JSONResponse({
                    "mcqs_raw": "\n\n".join(m["question_block"] for m in mapped_mcqs),
                    "mapped_mcqs": mapped_mcqs,
                    "txt_filename": txt_name,
                    "pdf_filename": pdf_name,
                    "json_filename": json_name,
                    "docx_filename": docx_name,
                    "txt_b64": txt_b64,
                    "pdf_b64": pdf_b64,
                    "docx_b64": docx_b64,
                    "json_b64": json_b64,
                })

        except Exception as e:
            logger.error(f"File saving failed: {e}")
            return error_response("Error saving output files", 500)

    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return error_response("Internal server error", 500)

    # Return response with base64 files
           



# ===================================================================
#                    DOWNLOAD ENDPOINT WITH COMPRESSION
# ===================================================================

@app.get("/download/{filename}")
async def download_file(filename: str, compress: bool = False):
    """
    Download generated file with optional compression
    Compression reduces bandwidth usage by 60-80%
    """
    safe_name = secure_filename(filename)
    path = os.path.join(RESULTS_FOLDER, safe_name)
    
    if not os.path.exists(path):
        logger.warning(f"File not found: {safe_name}")
        return JSONResponse({"error": "File not found"}, status_code=404)
    
    try:
        # Read file
        async with aiofiles.open(path, 'rb') as f:
            content = await f.read()
        
        # Compress if requested
        if compress:
            compressed = gzip.compress(content)
            logger.info(f"Compressed {safe_name}: {len(content)} -> {len(compressed)} bytes")
            
            return StreamingResponse(
                io.BytesIO(compressed),
                media_type="application/gzip",
                headers={
                    "Content-Disposition": f"attachment; filename={safe_name}.gz",
                    "Content-Encoding": "gzip"
                }
            )
        
        # Return uncompressed
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={safe_name}"}
        )
    
    except Exception as e:
        logger.error(f"Download failed for {safe_name}: {e}")
        return JSONResponse({"error": "Download failed"}, status_code=500)


# ===================================================================
#                         ADMIN ENDPOINTS
# ===================================================================

@app.get("/stats")
async def get_stats():
    """Get system statistics (for monitoring)"""
    return {
        "cache_size": mcq_cache.size(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/admin/clear-cache")
async def clear_cache():
    """Clear MCQ cache (admin only)"""
    mcq_cache.clear()
    logger.info("Cache cleared")
    return {"message": "Cache cleared successfully"}


# ===================================================================
#                         STARTUP/SHUTDOWN
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup"""
    logger.info("=" * 60)
    logger.info("AI MCQ Generator API Started")
    logger.info(f"Model: {settings.groq_model}")
    logger.info(f"Rate Limit: {settings.rate_limit_requests_per_minute}/min, {settings.rate_limit_requests_per_hour}/hour")
    logger.info(f"Max File Size: {settings.max_file_size_mb}MB")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown"""
    logger.info("AI MCQ Generator API Shutting Down")

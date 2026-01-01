"""
S3/MinIO Storage Service
========================

Handles file storage in S3-compatible storage (AWS S3 or MinIO).
"""

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, BinaryIO, AsyncIterator
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

from api.config import settings


@dataclass
class UploadResult:
    """Result of file upload."""
    success: bool
    s3_key: str
    size_bytes: int
    etag: Optional[str] = None
    error: Optional[str] = None


@dataclass 
class DownloadResult:
    """Result of file download."""
    success: bool
    local_path: str
    size_bytes: int
    error: Optional[str] = None


class StorageService:
    """
    Handles file storage in S3/MinIO.
    
    Features:
    - Presigned URLs for direct upload/download
    - Multipart upload for large files
    - Content type detection
    - File validation
    """
    
    ALLOWED_EXTENSIONS = {'.pt', '.pth', '.bin', '.safetensors', '.onnx'}
    MAX_FILE_SIZE = 50 * 1024 * 1024 * 1024  # 50GB
    
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name=settings.s3_region,
        )
        self.bucket = settings.s3_bucket
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                try:
                    self.client.create_bucket(Bucket=self.bucket)
                    print(f"Created bucket: {self.bucket}")
                except ClientError as create_error:
                    print(f"Could not create bucket: {create_error}")
    
    def generate_s3_key(self, user_id: str, filename: str, prefix: str = "models") -> str:
        """Generate unique S3 key for file."""
        ext = Path(filename).suffix.lower()
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        return f"{prefix}/{user_id}/{timestamp}_{unique_id}{ext}"
    
    async def get_presigned_upload_url(
        self,
        s3_key: str,
        content_type: str = "application/octet-stream",
        expires_in: int = 3600
    ) -> str:
        """
        Generate presigned URL for direct upload.
        
        Args:
            s3_key: The S3 key where file will be stored
            content_type: MIME type of the file
            expires_in: URL expiration in seconds (default 1 hour)
        
        Returns:
            Presigned upload URL
        """
        try:
            url = self.client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': s3_key,
                    'ContentType': content_type,
                },
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            raise Exception(f"Failed to generate upload URL: {e}")
    
    async def get_presigned_download_url(
        self,
        s3_key: str,
        expires_in: int = 3600,
        filename: Optional[str] = None
    ) -> str:
        """
        Generate presigned URL for download.
        
        Args:
            s3_key: The S3 key of the file
            expires_in: URL expiration in seconds
            filename: Optional filename for Content-Disposition header
        
        Returns:
            Presigned download URL
        """
        try:
            params = {
                'Bucket': self.bucket,
                'Key': s3_key,
            }
            
            if filename:
                params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'
            
            url = self.client.generate_presigned_url(
                'get_object',
                Params=params,
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            raise Exception(f"Failed to generate download URL: {e}")
    
    async def upload_file(
        self,
        file_path: str,
        s3_key: str,
        content_type: str = "application/octet-stream"
    ) -> UploadResult:
        """
        Upload file to S3.
        
        Args:
            file_path: Local path to file
            s3_key: Destination S3 key
            content_type: MIME type
        
        Returns:
            UploadResult with status
        """
        try:
            file_size = Path(file_path).stat().st_size
            
            # Use multipart upload for large files
            if file_size > 100 * 1024 * 1024:  # > 100MB
                return await self._multipart_upload(file_path, s3_key, content_type)
            
            with open(file_path, 'rb') as f:
                response = self.client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=f,
                    ContentType=content_type,
                )
            
            return UploadResult(
                success=True,
                s3_key=s3_key,
                size_bytes=file_size,
                etag=response.get('ETag'),
            )
            
        except Exception as e:
            return UploadResult(
                success=False,
                s3_key=s3_key,
                size_bytes=0,
                error=str(e),
            )
    
    async def _multipart_upload(
        self,
        file_path: str,
        s3_key: str,
        content_type: str,
        chunk_size: int = 100 * 1024 * 1024  # 100MB chunks
    ) -> UploadResult:
        """Upload large file using multipart upload."""
        try:
            file_size = Path(file_path).stat().st_size
            
            # Initiate multipart upload
            response = self.client.create_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                ContentType=content_type,
            )
            upload_id = response['UploadId']
            
            parts = []
            part_number = 1
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    part_response = self.client.upload_part(
                        Bucket=self.bucket,
                        Key=s3_key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=chunk,
                    )
                    
                    parts.append({
                        'PartNumber': part_number,
                        'ETag': part_response['ETag'],
                    })
                    part_number += 1
            
            # Complete multipart upload
            self.client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts},
            )
            
            return UploadResult(
                success=True,
                s3_key=s3_key,
                size_bytes=file_size,
            )
            
        except Exception as e:
            # Abort multipart upload on failure
            try:
                self.client.abort_multipart_upload(
                    Bucket=self.bucket,
                    Key=s3_key,
                    UploadId=upload_id,
                )
            except:
                pass
            
            return UploadResult(
                success=False,
                s3_key=s3_key,
                size_bytes=0,
                error=str(e),
            )
    
    async def download_file(
        self,
        s3_key: str,
        local_path: str
    ) -> DownloadResult:
        """
        Download file from S3.
        
        Args:
            s3_key: S3 key of file to download
            local_path: Local destination path
        
        Returns:
            DownloadResult with status
        """
        try:
            # Ensure directory exists
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.client.download_file(
                Bucket=self.bucket,
                Key=s3_key,
                Filename=local_path,
            )
            
            file_size = Path(local_path).stat().st_size
            
            return DownloadResult(
                success=True,
                local_path=local_path,
                size_bytes=file_size,
            )
            
        except Exception as e:
            return DownloadResult(
                success=False,
                local_path=local_path,
                size_bytes=0,
                error=str(e),
            )
    
    async def delete_file(self, s3_key: str) -> bool:
        """
        Delete file from S3.
        
        Args:
            s3_key: S3 key to delete
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            return True
        except Exception as e:
            print(f"Failed to delete {s3_key}: {e}")
            return False
    
    async def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            return True
        except ClientError:
            return False
    
    async def get_file_info(self, s3_key: str) -> Optional[dict]:
        """Get file metadata."""
        try:
            response = self.client.head_object(
                Bucket=self.bucket,
                Key=s3_key,
            )
            return {
                'size_bytes': response['ContentLength'],
                'content_type': response.get('ContentType'),
                'last_modified': response['LastModified'],
                'etag': response['ETag'],
            }
        except ClientError:
            return None
    
    def validate_file(self, filename: str, size_bytes: int) -> tuple:
        """
        Validate file before upload.
        
        Args:
            filename: Name of file
            size_bytes: Size in bytes
        
        Returns:
            (is_valid, error_message)
        """
        ext = Path(filename).suffix.lower()
        
        if ext not in self.ALLOWED_EXTENSIONS:
            return False, f"File type not allowed. Allowed: {self.ALLOWED_EXTENSIONS}"
        
        if size_bytes > self.MAX_FILE_SIZE:
            max_gb = self.MAX_FILE_SIZE / (1024 ** 3)
            return False, f"File too large. Maximum size: {max_gb:.0f}GB"
        
        return True, None


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service

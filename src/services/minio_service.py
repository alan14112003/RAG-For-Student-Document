import os
import logging
from pathlib import Path
from typing import Optional, BinaryIO
from datetime import timedelta
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinIOService:
    """Service để quản lý file storage với MinIO"""
    
    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket_name: str = "rag-documents",
        secure: bool = False,
    ):
        """
        Initialize MinIO client
        
        Args:
            endpoint: MinIO server endpoint (host:port)
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Default bucket name
            secure: Use HTTPS if True
        """
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        
        # Initialize MinIO client
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        
        # Ensure bucket exists
        self._ensure_bucket()
        
        logger.info(f"MinIO service initialized (endpoint={endpoint}, bucket={bucket_name})")
    
    def _ensure_bucket(self) -> None:
        """Tạo bucket nếu chưa tồn tại"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.debug(f"Bucket already exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            raise
    
    def upload_file(
        self,
        file_path: str | Path,
        object_name: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload file từ local path lên MinIO
        
        Args:
            file_path: Đường dẫn file local
            object_name: Tên object trong MinIO (mặc định dùng filename)
            content_type: MIME type của file
            
        Returns:
            object_name: Tên object đã upload
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Default object name
        if object_name is None:
            object_name = file_path.name
        
        try:
            # Upload file
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=str(file_path),
                content_type=content_type,
            )
            
            logger.info(f"Uploaded file to MinIO: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"Failed to upload file to MinIO: {e}")
            raise
    
    def upload_fileobj(
        self,
        file_data: BinaryIO,
        object_name: str,
        length: int,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload file từ file object (memory) lên MinIO
        
        Args:
            file_data: File-like object (bytes)
            object_name: Tên object trong MinIO
            length: Kích thước file (bytes)
            content_type: MIME type
            
        Returns:
            object_name: Tên object đã upload
        """
        try:
            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=file_data,
                length=length,
                content_type=content_type,
            )
            
            logger.info(f"Uploaded file object to MinIO: {object_name}")
            return object_name
            
        except S3Error as e:
            logger.error(f"Failed to upload file object to MinIO: {e}")
            raise
    
    def download_file(
        self,
        object_name: str,
        file_path: str | Path,
    ) -> Path:
        """
        Download file từ MinIO về local
        
        Args:
            object_name: Tên object trong MinIO
            file_path: Đường dẫn lưu file local
            
        Returns:
            file_path: Đường dẫn file đã download
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=str(file_path),
            )
            
            logger.info(f"Downloaded file from MinIO: {object_name}")
            return file_path
            
        except S3Error as e:
            logger.error(f"Failed to download file from MinIO: {e}")
            raise

    def get_object_bytes(self, object_name: str) -> bytes:
        """
        Read entire object content into memory.

        Args:
            object_name: Tên object trong MinIO

        Returns:
            Raw bytes của object
        """
        try:
            response = self.client.get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()

            logger.debug(f"Fetched object bytes from MinIO: {object_name}")
            return data
        except S3Error as e:
            logger.error(f"Failed to fetch object bytes from MinIO: {e}")
            raise
    
    def get_file_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=1),
    ) -> str:
        """
        Tạo presigned URL để download file
        
        Args:
            object_name: Tên object trong MinIO
            expires: Thời gian hết hạn URL
            
        Returns:
            url: Presigned URL
        """
        try:
            url = self.client.presigned_get_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                expires=expires,
            )
            
            logger.debug(f"Generated presigned URL for: {object_name}")
            return url
            
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise
    
    def delete_file(self, object_name: str) -> None:
        """
        Xóa file khỏi MinIO
        
        Args:
            object_name: Tên object trong MinIO
        """
        try:
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )
            
            logger.info(f"Deleted file from MinIO: {object_name}")
            
        except S3Error as e:
            logger.error(f"Failed to delete file from MinIO: {e}")
            raise
    
    def file_exists(self, object_name: str) -> bool:
        """
        Kiểm tra file có tồn tại trong MinIO không
        
        Args:
            object_name: Tên object trong MinIO
            
        Returns:
            exists: True nếu file tồn tại
        """
        try:
            self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )
            return True
        except S3Error:
            return False
    
    def get_file_info(self, object_name: str) -> dict:
        """
        Lấy thông tin file từ MinIO
        
        Args:
            object_name: Tên object trong MinIO
            
        Returns:
            info: Dict chứa thông tin file
        """
        try:
            stat = self.client.stat_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
            )
            
            return {
                "object_name": stat.object_name,
                "size": stat.size,
                "etag": stat.etag,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "metadata": stat.metadata,
            }
            
        except S3Error as e:
            logger.error(f"Failed to get file info from MinIO: {e}")
            raise
    
    def list_files(self, prefix: Optional[str] = None) -> list[str]:
        """
        List tất cả files trong bucket
        
        Args:
            prefix: Lọc theo prefix (ví dụ: "user_123/")
            
        Returns:
            files: List tên objects
        """
        try:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix,
                recursive=True,
            )
            
            return [obj.object_name for obj in objects]
            
        except S3Error as e:
            logger.error(f"Failed to list files from MinIO: {e}")
            raise


def get_minio_service() -> MinIOService:
    """Factory function để tạo MinIO service từ environment variables"""
    return MinIOService(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        bucket_name=os.getenv("MINIO_BUCKET", "rag-documents"),
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )

import os
import json
import uuid
import logging
import typing as t
from datetime import datetime, timedelta, timezone
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings

from openai import OpenAI
from openai._models import FinalRequestOptions
from openai._types import Omit
from openai._utils import is_given

fabric_base_url = os.getenv("FABRIC_BASE_URL")

class AzureBlobManager:
    def __init__(self, connection_string: str):
        self.set_connection_string(connection_string)

    def set_connection_string(self, connection_string: str) -> None:
        self.connection_string = connection_string
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    def _get_blob_client(self, container_name: str, blob_path: str):
        return self.blob_service_client.get_blob_client(container=container_name, blob=blob_path)
    
    def _read_blob_content(self, blob_client) -> bytes:
        stream = blob_client.download_blob()
        return stream.readall()
    
    def _generate_sas_token(self, container_name: str, blob_name: str, expiry_hours: int = 24) -> str:
        return generate_blob_sas(
            account_name=self.blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=self.blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
        )
    
    def read_credential_blob(self, container_name: str = "credentials", file_name: str = "token.json"):
        blob_client = self._get_blob_client(container_name, file_name)
        try:
            content = self._read_blob_content(blob_client)
            return json.loads(content.decode('utf-8'))['token']
        except Exception as e:
            logging.error(f"Failed to read credential blob: {e}")
            raise

    def upload_df_blob(self, data: str, container_name: str = "tables"):
       try:
           table_id = str(uuid.uuid4())
           file_name = f"{table_id}.md"
           blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=file_name)
           blob_client.upload_blob(data, overwrite=True)
           return table_id
       except Exception as e:
           print(f"Failed to write data blob: {e}")
           raise
    
    def read_df_blob(self, table_id: str, container_name: str = "tables"):
        try:
            file_name = f"{table_id}.md"
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=file_name)
            content = self._read_blob_content(blob_client)
            return content.decode('utf-8')
        except Exception as e:
            print(f"Failed to read data blob: {e}")
            raise

    def upload_blob(self, image_data: bytes, file_name: str, content_type: str, container_name: str = "files", expiry_hours: int = 3500):
        blob_client = self._get_blob_client(container_name, file_name)
        try:
            blob_client.upload_blob(image_data, overwrite=True, content_settings=ContentSettings(content_type=content_type))
            logging.info(f"File uploaded successfully to {container_name}/{file_name}")
            sas_token = self._generate_sas_token(container_name, file_name, expiry_hours)
            blob_url = f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{container_name}/{file_name}?{sas_token}"
            return blob_url
        except Exception as e:
            logging.error(f"Failed to upload image blob: {e}")
            raise
    
    def read_thread(self, user_id: str, container_name: str = "credentials"):
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                print(f"Container {container_name} does not exist. Creating it...")
                container_client.create_container()
                
            file_name = "threads.json"
            blob_client = self._get_blob_client(container_name, file_name)
            
            try:
                content = self._read_blob_content(blob_client)
                threads = json.loads(content.decode('utf-8'))
                return threads.get(user_id, None)
            except Exception as e:
                blob_client.upload_blob(json.dumps({}), overwrite=True)
                return None
        except Exception as e:
            print(f"Error in read_thread: {e}")
            return None
    
    def write_thread(self, user_id: str, thread_id: str, container_name: str = "credentials"):
        file_name = "threads.json"
        blob_client = self._get_blob_client(container_name, file_name)
        try:
            content = self._read_blob_content(blob_client)
            threads = json.loads(content.decode('utf-8'))
            threads[user_id] = thread_id
            blob_client.upload_blob(json.dumps(threads), overwrite=True)
        except Exception as e:
            logging.error(f"Failed to write thread blob: {e}")
            raise

class FabricOpenAI(OpenAI):
    def __init__(
        self,
        bearer_token: str,
        api_version: str ="2024-05-01-preview",
        **kwargs: t.Any,
    ) -> None:
        self.api_version = api_version
        self.bearer_token = bearer_token
        default_query = kwargs.pop("default_query", {})
        default_query["api-version"] = self.api_version
        print("Fabric base url: ", fabric_base_url)
        super().__init__(
            api_key="",
            base_url=fabric_base_url,
            default_query=default_query,
            **kwargs,
        )
    
    def _prepare_options(self, options: FinalRequestOptions) -> None:
        headers: dict[str, str | Omit] = (
            {**options.headers} if is_given(options.headers) else {}
        )
        options.headers = headers
        headers["Authorization"] = f"Bearer {self.bearer_token}"
        if "Accept" not in headers:
            headers["Accept"] = "application/json"
        if "ActivityId" not in headers:
            correlation_id = str(uuid.uuid4())
            headers["ActivityId"] = correlation_id

        return super()._prepare_options(options)
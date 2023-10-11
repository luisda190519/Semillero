from azure.storage.blob import BlobServiceClient
import os
import shutil

connection_string = "DefaultEndpointsProtocol=https;AccountName=modelosemillero;AccountKey=tyBROI5tn6oxVpk1K/nKL8muX9ZKy8OsMHplkLtyyyNDC4vG3QSxWw22uIDiMthDuYWNSEX006un+AStCP3YHw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "semillero"
model_height = "model_ep_48.pth.tar"
model_weight = "model_ep_37.pth.tar"

blob_client1 = blob_service_client.get_blob_client(container_name, model_height)
blob_client2 = blob_service_client.get_blob_client(container_name, model_weight)

with open(model_height, "wb") as model_file:
    download_stream = blob_client1.download_blob()
    model_file.write(download_stream.readall())    

model_path = os.path.join("models", model_height)
shutil.move(model_height, model_path)

with open(model_weight, "wb") as model_file:
    download_stream = blob_client2.download_blob()
    model_file.write(download_stream.readall())   

model_path = os.path.join("models", model_weight)
shutil.move(model_weight, model_path)
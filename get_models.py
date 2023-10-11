from azure.storage.blob import BlobServiceClient
import os
import shutil

connection_string = "DefaultEndpointsProtocol=https;AccountName=modelosemillero;AccountKey=tyBROI5tn6oxVpk1K/nKL8muX9ZKy8OsMHplkLtyyyNDC4vG3QSxWw22uIDiMthDuYWNSEX006un+AStCP3YHw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "semillero"
model_height = "model_ep_48.pth.tar"
model_weight = "model_ep_37.pth.tar"

model_directory = "models"
os.makedirs(model_directory, exist_ok=True)

model_height_path = os.path.join(model_directory, model_height)
model_weight_path = os.path.join(model_directory, model_weight)

if not (os.path.exists(model_height_path) and os.path.exists(model_weight_path)):
    blob_client1 = blob_service_client.get_blob_client(container_name, model_height)
    blob_client2 = blob_service_client.get_blob_client(container_name, model_weight)

    with open(model_height, "wb") as model_file:
        download_stream = blob_client1.download_blob()
        model_file.write(download_stream.readall())

    shutil.move(model_height, model_height_path)

    with open(model_weight, "wb") as model_file:
        download_stream = blob_client2.download_blob()
        model_file.write(download_stream.readall())

    shutil.move(model_weight, model_weight_path)

print("Models are downloaded and stored in the 'models' directory.")

import os
import urllib.request

def download_hed_model():
    # Create models directory if it doesn't exist
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # URLs for the model files
    model_urls = {
        "deploy.prototxt": "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt",
        "hed_pretrained_bsds.caffemodel": "http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
    }

    # Download files
    for filename, url in model_urls.items():
        output_path = os.path.join(model_dir, filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, output_path)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                return False
        else:
            print(f"{filename} already exists")
    
    return True

if __name__ == "__main__":
    download_hed_model()

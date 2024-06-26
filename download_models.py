import requests
from tqdm import tqdm
import os

def download_file(url, filename):
    """Download a file from a given URL and save it with the specified filename."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)

def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    model_names = ['0n', '5n', '50n', '80n']
    base_url = "https://huggingface.co/Zual/chessGPT/resolve/main/ckpt_{}8l240000iter.pt"

    for model in model_names:
        url = base_url.format(model)
        filename = os.path.join('models', f"ckpt_{model}8l240000iter.pt")
        
        print(f"Downloading model: ckpt_{model}8l240000iter.pt")
        try:
            download_file(url, filename)
            print(f"Successfully downloaded: {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")

    print("All downloads completed.")

if __name__ == "__main__":
    main()
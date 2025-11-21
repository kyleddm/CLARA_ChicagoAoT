import requests
import csv

urls=[]
with open('sensors.csv',mode='r') as senFil:
    csv_reader = csv.DictReader(senFil)
    for row in csv_reader:
        urls.append(row['datasheet'])

# Download each file
for url in urls:
    file_name = url.split("/")[-1]  # Extract the file name from the URL
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP request errors
        with open('datasheets/'+file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {file_name}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
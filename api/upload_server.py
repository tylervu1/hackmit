import requests
import sys

def send_file(file_path, target_ip, target_port):
    url = f"http://{target_ip}:{target_port}/upload"
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.split('/')[-1], f)}
        r = requests.post(url, files=files)
        print(r.text)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python send_file.py <file_path> <target_ip> <target_port>")
    else:
        file_path = sys.argv[1]
        target_ip = sys.argv[2]
        target_port = sys.argv[3]
        send_file(file_path, target_ip, target_port)

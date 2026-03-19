import requests
import xml.etree.ElementTree as ET
from datetime import datetime

# Fetch all MPs
url = "https://api.parliament.uk/mnis-prodder/parse"
params = {
    'filter': 'membership=all|house=commons',
    'include': 'constituencies'
}

response = requests.get(url, params=params, timeout=60)
root = ET.fromstring(response.content)

# Parse members
mps_1990_2024 = []
for member in root.findall('.//Member'):
    name = member.find('DisplayAs').text
    gender = member.find('Gender').text if member.find('Gender') is not None else None
    
    # Check constituencies for 1990-2024 service
    for const in member.findall('.//Constituency'):
        start_date = const.find('StartDate').text
        end_date = const.find('EndDate').text if const.find('EndDate') is not None else None
        
        # Check if served between 1990-2024
        start = datetime.fromisoformat(start_date.split('T')[0])
        end = datetime.fromisoformat(end_date.split('T')[0]) if end_date else datetime.now()
        
        if start <= datetime(2024, 12, 31) and end >= datetime(1990, 1, 1):
            mps_1990_2024.append({'name': name, 'gender': gender})
            break

print(f"Found {len(mps_1990_2024)} MPs who served 1990-2024")
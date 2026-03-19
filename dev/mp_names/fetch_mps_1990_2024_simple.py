"""
Simple UK MPs Name and Gender Fetcher (1990-2024)
==================================================
Fetches MP names and gender for those who served between 1990-2024
"""

import requests
import pandas as pd
from datetime import datetime

# API endpoint
API_URL = "https://members-api.parliament.uk/api/Members/Search"

print("Fetching all MPs from Parliament API (this may take a moment)...")

# We need to paginate through all results
all_mps = []
skip = 0
take = 50

while True:
    params = {
        "House": 1,  # Commons
        "IsCurrentMember": False,  # Get all members, not just current
        "skip": skip,
        "take": take
    }
    
    response = requests.get(API_URL, params=params, timeout=10)
    data = response.json()
    items = data.get('items', [])
    
    if not items:
        break
    
    all_mps.extend(items)
    print(f"Fetched {len(all_mps)} MPs...")
    
    # Check if we've got all results
    if len(all_mps) >= data.get('totalResults', 0):
        break
    
    skip += take

print(f"Total MPs fetched: {len(all_mps)}")

# Filter for MPs who served between 1990-2024
filtered_mps = []
start_date = datetime(1990, 1, 1)
end_date = datetime(2024, 12, 31)

for member in all_mps:
    value = member.get('value', {})
    membership = value.get('latestHouseMembership', {})
    
    # Get dates
    start_str = membership.get('membershipStartDate')
    end_str = membership.get('membershipEndDate')
    
    # Check if MP served during 1990-2024
    include = False
    
    if start_str:
        try:
            start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
            if end_str:
                end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                if start <= end_date and end >= start_date:
                    include = True
            else:
                # No end date means still serving
                if start <= end_date:
                    include = True
        except:
            pass
    
    if include:
        filtered_mps.append({
            'id': value.get('id'),
            'name_display': value.get('nameDisplayAs'),
            'name_full_title': value.get('nameFullTitle'),
            'name_list': value.get('nameListAs'),
            'name_address': value.get('nameAddressAs'),
            'gender': value.get('gender')
        })

# Remove duplicates by name (same person may have multiple terms)
df = pd.DataFrame(filtered_mps)
df = df.drop_duplicates(subset=['name_display'], keep='first')

# Save to CSV
df.to_csv('mps_1990_2024.csv', index=False)

print(f"\n✓ Saved {len(df)} unique MPs (1990-2024) to mps_1990_2024.csv")
print(f"\nGender distribution:")
print(df['gender'].value_counts())

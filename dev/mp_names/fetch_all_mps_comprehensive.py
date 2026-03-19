"""
UK Parliament MPs Data Fetcher (1990-2024) - COMPREHENSIVE VERSION
===================================================================
This script fetches ALL MPs who served between 1990 and 2024 by:
1. Getting all members from the search endpoint
2. Fetching detailed history for each member to get ALL their memberships
3. Filtering based on whether ANY of their memberships overlapped with 1990-2024

This will give us a much more complete list than just looking at latest membership.

Requirements:
    pip install requests pandas

Usage:
    python fetch_all_mps_comprehensive.py
"""

import requests
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Set

def fetch_all_member_ids() -> List[int]:
    """
    Get all member IDs from the Members API.
    """
    base_url = "https://members-api.parliament.uk/api/Members/Search"
    
    params = {
        "House": 1,  # Commons only
        "IsCurrentMember": False,  # Get all members
        "skip": 0,
        "take": 50
    }
    
    all_ids = []
    
    print("Step 1: Fetching all member IDs...")
    
    while True:
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            if not items:
                break
            
            for member in items:
                member_id = member.get('value', {}).get('id')
                if member_id:
                    all_ids.append(member_id)
            
            print(f"Found {len(all_ids)} member IDs...")
            
            # Check if we've got all results
            if params['skip'] + params['take'] >= data.get('totalResults', 0):
                break
            
            params['skip'] += params['take']
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"Total member IDs found: {len(all_ids)}\n")
    return all_ids

def fetch_member_details(member_id: int) -> Dict:
    """
    Fetch detailed information for a specific member, including ALL constituencies.
    """
    url = f"https://members-api.parliament.uk/api/Members/{member_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('value', {})
    except:
        return {}

def check_member_served_in_period(member_details: Dict, start_date: datetime, end_date: datetime) -> bool:
    """
    Check if a member served during the specified period based on ANY of their memberships.
    """
    # Check latest membership
    latest = member_details.get('latestHouseMembership', {})
    
    if latest:
        start_str = latest.get('membershipStartDate')
        end_str = latest.get('membershipEndDate')
        
        if start_str:
            try:
                membership_start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                membership_end = datetime.fromisoformat(end_str.replace('Z', '+00:00')) if end_str else datetime.now()
                
                # Check if this membership overlaps with our date range
                if membership_start <= end_date and membership_end >= start_date:
                    return True
            except:
                pass
    
    # Also check house memberships array if available
    house_memberships = member_details.get('houseMemberships', [])
    for membership in house_memberships:
        start_str = membership.get('membershipStartDate')
        end_str = membership.get('membershipEndDate')
        
        if start_str:
            try:
                membership_start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                membership_end = datetime.fromisoformat(end_str.replace('Z', '+00:00')) if end_str else datetime.now()
                
                if membership_start <= end_date and membership_end >= start_date:
                    return True
            except:
                pass
    
    return False

def fetch_mps_1990_2024() -> List[Dict]:
    """
    Main function to fetch all MPs who served between 1990-2024.
    """
    start_date = datetime(1990, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Step 1: Get all member IDs
    member_ids = fetch_all_member_ids()
    
    # Step 2: Fetch details for each member and filter
    print("Step 2: Fetching detailed information for each member...")
    print("This will take several minutes...\n")
    
    qualifying_mps = []
    processed = 0
    
    for member_id in member_ids:
        processed += 1
        
        if processed % 50 == 0:
            print(f"Processed {processed}/{len(member_ids)} members... Found {len(qualifying_mps)} qualifying MPs")
        
        # Fetch detailed info
        details = fetch_member_details(member_id)
        
        if not details:
            continue
        
        # Check if they served in 1990-2024
        if check_member_served_in_period(details, start_date, end_date):
            mp_data = {
                'id': details.get('id'),
                'name_display': details.get('nameDisplayAs'),
                'name_full_title': details.get('nameFullTitle'),
                'name_list': details.get('nameListAs'),
                'gender': details.get('gender'),
                'party': details.get('latestParty', {}).get('name') if details.get('latestParty') else None
            }
            qualifying_mps.append(mp_data)
        
        # Be polite to the API
        time.sleep(0.1)
    
    print(f"\nTotal MPs who served 1990-2024: {len(qualifying_mps)}")
    return qualifying_mps

def save_to_csv(members_data: List[Dict], filename: str = 'mps_1990_2024_complete.csv',
                minimal: bool = False) -> None:
    """
    Save MP data to CSV.
    """
    if not members_data:
        print("No data to save!")
        return
    
    df = pd.DataFrame(members_data)
    
    # Remove duplicates by ID
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    if minimal:
        df = df[['name_display', 'gender']]
        df.columns = ['name', 'gender']
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n✓ Data saved to {filename}")
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique MPs: {len(df)}")
    
    if 'gender' in df.columns:
        print(f"\nGender distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    if not minimal and 'party' in df.columns:
        print(f"\nTop 10 parties:")
        party_counts = df['party'].value_counts().head(10)
        for party, count in party_counts.items():
            print(f"  {party}: {count}")
    
    print(f"\n{'='*60}")
    print("First 10 rows:")
    print(df.head(10).to_string())

def main():
    """Main execution"""
    print("="*60)
    print("UK Parliament MPs Data Fetcher (1990-2024)")
    print("COMPREHENSIVE VERSION - Fetches complete membership history")
    print("="*60)
    print()
    
    mps = fetch_mps_1990_2024()
    
    if mps:
        # Full dataset
        save_to_csv(mps, filename='uk_mps_1990_2024_complete.csv', minimal=False)
        
        # Minimal dataset
        save_to_csv(mps, filename='uk_mps_1990_2024_names_gender.csv', minimal=True)
        
        print(f"\n✓ Complete! Files created:")
        print(f"  - uk_mps_1990_2024_complete.csv")
        print(f"  - uk_mps_1990_2024_names_gender.csv")
    else:
        print("\n✗ Failed to fetch data")

if __name__ == "__main__":
    main()

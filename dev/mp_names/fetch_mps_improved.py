"""
UK Parliament MPs Data Fetcher
================================
This script fetches current UK Members of Parliament (MPs) data from the 
UK Parliament Members API and saves their names and gender to a CSV file.

Requirements:
    pip install requests pandas

API Documentation:
    https://members-api.parliament.uk/index.html

Usage:
    python fetch_mps.py
"""

import requests
import pandas as pd
import time
from typing import List, Dict, Optional

def fetch_all_mps(house: int = 1, current_only: bool = True) -> List[Dict]:
    """
    Fetch MPs from the UK Parliament Members API.
    
    Args:
        house: 1 for Commons, 2 for Lords
        current_only: If True, fetch only current members
        
    Returns:
        List of dictionaries containing MP data
    """
    base_url = "https://members-api.parliament.uk/api/Members/Search"
    
    params = {
        "House": house,
        "IsCurrentMember": current_only,
        "skip": 0,
        "take": 20  # Items per page
    }
    
    all_members = []
    
    print("Fetching MPs from Parliament API...")
    print(f"House: {'Commons' if house == 1 else 'Lords'}")
    print(f"Current members only: {current_only}\n")
    
    while True:
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            if not items:
                break
            
            # Extract relevant fields from each member
            for member in items:
                value = member.get('value', {})
                
                mp_data = {
                    'id': value.get('id'),
                    'name_display': value.get('nameDisplayAs'),
                    'name_full_title': value.get('nameFullTitle'),
                    'name_list': value.get('nameListAs'),
                    'name_address': value.get('nameAddressAs'),
                    'gender': value.get('gender'),
                    'party': value.get('latestParty', {}).get('name') if value.get('latestParty') else None,
                    'thumbnail_url': value.get('thumbnailUrl')
                }
                all_members.append(mp_data)
            
            print(f"Fetched {len(all_members)} members so far...")
            
            # Check if there are more results
            total_results = data.get('totalResults', 0)
            if len(all_members) >= total_results:
                break
            
            # Update skip for next page
            params['skip'] += params['take']
            
            # Be polite to the API - add a small delay
            time.sleep(0.3)
            
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching data: {e}")
            print("\nNote: If you're getting connection errors, make sure:")
            print("  1. You have an active internet connection")
            print("  2. The API endpoint is accessible")
            print("  3. You have the required packages installed (requests, pandas)")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            break
    
    print(f"\nTotal members fetched: {len(all_members)}")
    return all_members

def save_to_csv(members_data: List[Dict], filename: str = 'uk_mps.csv', 
                columns: Optional[List[str]] = None) -> None:
    """
    Save MP data to a CSV file using pandas.
    
    Args:
        members_data: List of member dictionaries
        filename: Output CSV filename
        columns: List of columns to include (None = all columns)
    """
    if not members_data:
        print("No data to save!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(members_data)
    
    # Select specific columns if requested
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        df = df[available_cols]
    
    # Save to CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n✓ Data saved to {filename}")
    
    # Display statistics
    print(f"\n{'='*50}")
    print(f"DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Total records: {len(df)}")
    
    if 'gender' in df.columns:
        print(f"\nGender distribution:")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    if 'party' in df.columns:
        print(f"\nTop 5 parties by member count:")
        party_counts = df['party'].value_counts().head(5)
        for party, count in party_counts.items():
            print(f"  {party}: {count}")
    
    print(f"\n{'='*50}")
    print(f"SAMPLE DATA (first 5 rows)")
    print(f"{'='*50}")
    print(df.head().to_string())
    print()

def main():
    """Main execution function"""
    
    # Fetch all current MPs from House of Commons
    mps = fetch_all_mps(house=1, current_only=True)
    
    if mps:
        # Save full dataset
        save_to_csv(mps, filename='uk_mps_full.csv')
        
        # Save minimal dataset with just names and gender
        save_to_csv(
            mps, 
            filename='uk_mps_names_gender.csv',
            columns=['name_display', 'gender']
        )
        
        print(f"\n✓ Complete! Check the output files:")
        print(f"  - uk_mps_full.csv (all fields)")
        print(f"  - uk_mps_names_gender.csv (names and gender only)")
    else:
        print("\n✗ Failed to fetch data. Please check the error messages above.")

if __name__ == "__main__":
    main()

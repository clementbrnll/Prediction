import aiohttp
import asyncio
import json
from datetime import datetime, timedelta

async def fetch_data(session, station_id, start_date, end_date):
    url = f"https://portail-api-data.montpellier3m.fr/bikestation_timeseries/urn%3Angsi-ld%3Astation%3A{station_id:03d}/attrs/availableBikeNumber"
    params = {
        'fromDate': start_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'toDate': end_date.strftime('%Y-%m-%dT%H:%M:%S')
    }
    async with session.get(url, params=params) as response:
        if response.status == 200:
            return await response.json(), None
        else:
            error_message = f"Error fetching data for station {station_id} from {start_date} to {end_date}: {response.status}"
            print(error_message)
            return None, error_message

async def fetch_all_data_for_station(session, station_id):
    start_date = datetime(2023, 1, 4)
    end_date = datetime(2024, 1, 4)
    delta = timedelta(days=7)

    all_data = []
    current_start = start_date
    errors = []

    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        data, error = await fetch_data(session, station_id, current_start, current_end)
        if data:
            all_data.extend(data)
        elif error:
            errors.append(error)
        current_start = current_end + timedelta(seconds=1)
    
    return all_data, errors

async def main():
    async with aiohttp.ClientSession() as session:
        start_date = datetime(2023, 1, 4)
        end_date = datetime(2024, 1, 4)
        valid_stations = []

        # Identification des stations valides
        for station_id in range(1, 62):
            data, error = await fetch_data(session, station_id, start_date, end_date)
            if data:
                print(f"Station ID {station_id:03d} is valid.")
                valid_stations.append(station_id)
            else:
                print(f"Station ID {station_id:03d} is not valid or has no data.")

        print("Valid stations:", valid_stations)
        
        tasks = []
        all_stations_data = {}
        all_errors = {}
        
        # Collecte des données pour les stations valides
        for station_id in valid_stations:
            tasks.append(fetch_all_data_for_station(session, station_id))

        results = await asyncio.gather(*tasks)

        for station_id, (station_data, errors) in zip(valid_stations, results):
            if station_data:
                all_stations_data[f"station_{station_id:03d}"] = station_data
            if errors:
                all_errors[f"station_{station_id:03d}"] = errors

        # Enregistrement des données dans un fichier JSON
        with open('all_stations_data.json', 'w') as f:
            json.dump(all_stations_data, f, indent=4)

        # Enregistrement des erreurs dans un fichier JSON
        with open('errors.json', 'w') as f:
            json.dump(all_errors, f, indent=4)

if __name__ == "__main__":
    asyncio.run(main())

"""Download the two authorized public datasets only; logs remain concise."""
from concurrent.futures import ThreadPoolExecutor
from src.data.screening_io import download_source

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download_source,['openr1','code']))

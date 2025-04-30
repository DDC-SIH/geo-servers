import requests
import time
import json
import statistics
import argparse
from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# # Run benchmark with default settings (5 iterations per endpoint)
# python benchmark.py

# # Run benchmark with more iterations
# python benchmark.py --iterations 10

# # Run benchmark for specific endpoints only
# python benchmark.py --endpoints generate-gif cloud-mask

# # Save benchmark results to a JSON file
# python benchmark.py --output benchmark_results.json

# # Generate performance chart
# python benchmark.py --chart performance.png

# # Show verbose output during benchmarking
# python benchmark.py --verbose

# API base URL
BASE_URL = "http://74.226.242.56:5000"

# Test data for each endpoint
TEST_DATA = {
    "generate-gif": {
        "SatelliteId": "3R",
        "startDateTime": "2025-03-22T00:00:00",
        "endDateTime": "2025-03-22T00:05:45",
        "interval": 1,
        "processingLevel": "L1C",
        "productType": "ASIA_MER",
        "bandName": "TIR2",
        "colourmap": "viridis",
        "bbox": {
            "minx": 72.5,
            "miny": 15.5,
            "maxx": 88.5,
            "maxy": 27.5
        }
    },
    "download/raw": [
        "http://74.226.242.56:8000/cog/bbox/72.0,15.0,78.0,25.0.tif?url=/path/to/file1.cog.tif&bidx=1&bidx=2&bidx=3",
        "http://74.226.242.56:8000/cog/bbox/72.0,15.0,78.0,25.0.tif?url=/path/to/file2.cog.tif&bidx=1&bidx=2&bidx=3"
    ],
    "download/layered": {
        "format": "png",
        "data": [
            {
                "directURL": "http://74.226.242.56:8000/cog/bbox/72.0,15.0,78.0,25.0.tif?url=/path/to/file1.cog.tif&bidx=1&bidx=2&bidx=3",
                "zIndex": 0,
                "transparency": 0.8
            },
            {
                "directURL": "http://74.226.242.56:8000/cog/bbox/72.0,15.0,78.0,25.0.tif?url=/path/to/file2.cog.tif&bidx=1&bidx=2&bidx=3",
                "zIndex": 1,
                "transparency": 0.6
            }
        ]
    },
    "pointprobe": {
        "SatelliteId": "3R",
        "startDateTime": "2025-03-22T00:00:00",
        "endDateTime": "2025-03-22T06:00:00",
        "interval": 1,
        "processingLevel": "L1C",
        "productType": "ASIA_MER",
        "bandName": "TIR2",
        "coordinate": {
            "lat": 22.5726,
            "lon": 88.3639
        }
    },
    "cloud-mask": {
        "SatelliteId": "3R",
        "startDateTime": "2025-03-22T00:00:00",
        "endDateTime": "2025-03-22T00:05:45",
        "processingLevel": "L1C",
        "productType": "ASIA_MER",
        "bbox": {
            "minx": 72.5,
            "miny": 15.5,
            "maxx": 88.5,
            "maxy": 27.5
        },
        "format": "json",
        "confidenceThreshold": 0.5,
        "maskColor": [255, 0, 0],
        "opacity": 0.5
    }
}

def run_benchmark(endpoint, iterations=5, timeout=120, verbose=False):
    """
    Benchmark a specific API endpoint
    
    Args:
        endpoint (str): API endpoint to benchmark
        iterations (int): Number of test iterations
        timeout (int): Request timeout in seconds
        verbose (bool): Whether to print detailed output
        
    Returns:
        dict: Benchmark results
    """
    url = f"{BASE_URL}/{endpoint}"
    results = []
    successful = 0
    failed = 0
    error_messages = []
    
    print(f"\nBenchmarking endpoint: {endpoint}")
    
    # For displaying progress
    with tqdm(total=iterations, desc="Running tests") as pbar:
        for i in range(iterations):
            try:
                # Get the appropriate test data for this endpoint
                data = TEST_DATA.get(endpoint, {})
                
                # Start timer
                start_time = time.time()
                
                # Send the request
                if endpoint in ["download/raw", "download/layered", "generate-gif", "pointprobe", "cloud-mask"]:
                    response = requests.post(url, json=data, timeout=timeout)
                else:
                    response = requests.get(url, timeout=timeout)
                
                # End timer
                end_time = time.time()
                
                # Calculate duration
                duration = end_time - start_time
                
                # Check status code
                if response.status_code == 200:
                    successful += 1
                    if verbose:
                        print(f"  ✓ Request {i+1} successful: {duration:.4f} seconds")
                else:
                    failed += 1
                    error_msg = f"Status {response.status_code}: {response.text}"
                    error_messages.append(error_msg)
                    if verbose:
                        print(f"  ✗ Request {i+1} failed: {error_msg}")
                
                results.append(duration)
            
            except Exception as e:
                duration = time.time() - start_time
                failed += 1
                error_msg = str(e)
                error_messages.append(error_msg)
                if verbose:
                    print(f"  ✗ Request {i+1} failed: {error_msg}")
                results.append(duration)
            
            pbar.update(1)
    
    # Calculate statistics
    if results:
        stats = {
            "min": min(results),
            "max": max(results),
            "avg": statistics.mean(results),
            "median": statistics.median(results),
            "p95": np.percentile(results, 95) if len(results) >= 1 else None,
            "successful": successful,
            "failed": failed,
            "total": iterations,
            "success_rate": (successful / iterations) * 100,
            "error_messages": error_messages
        }
        try:
            stats["stdev"] = statistics.stdev(results) if len(results) > 1 else 0
        except:
            stats["stdev"] = 0
    else:
        stats = {
            "min": None, "max": None, "avg": None, "median": None, "p95": None,
            "stdev": None, "successful": 0, "failed": iterations, "total": iterations,
            "success_rate": 0, "error_messages": error_messages
        }
    
    return stats

def plot_results(all_results, output_file=None):
    """Generate bar chart of response times for each endpoint"""
    endpoints = list(all_results.keys())
    avg_times = [results["avg"] for results in all_results.values()]
    min_times = [results["min"] for results in all_results.values()]
    max_times = [results["max"] for results in all_results.values()]
    p95_times = [results["p95"] for results in all_results.values()]
    
    x = np.arange(len(endpoints))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.bar(x - 1.5*width, min_times, width, label='Min')
    ax.bar(x - 0.5*width, avg_times, width, label='Avg')
    ax.bar(x + 0.5*width, p95_times, width, label='95th %ile')
    ax.bar(x + 1.5*width, max_times, width, label='Max')
    
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('API Endpoint Performance Benchmark')
    ax.set_xticks(x)
    ax.set_xticklabels(endpoints, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Performance chart saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Benchmark the geo-servers API endpoints")
    parser.add_argument("--endpoints", nargs="+", default=["generate-gif", "download/raw", "download/layered", "pointprobe", "cloud-mask"], 
                        help="Endpoints to benchmark")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per endpoint")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("--output", type=str, help="Output report filename")
    parser.add_argument("--chart", type=str, help="Output chart filename")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting benchmark at {timestamp}")
    print(f"Base URL: {BASE_URL}")
    print(f"Iterations per endpoint: {args.iterations}")
    
    all_results = {}
    
    for endpoint in args.endpoints:
        results = run_benchmark(endpoint, args.iterations, args.timeout, args.verbose)
        all_results[endpoint] = results
    
    # Display results in tabular format
    table_data = []
    for endpoint, stats in all_results.items():
        table_data.append([
            endpoint,
            f"{stats['min']:.4f}" if stats['min'] else "N/A",
            f"{stats['avg']:.4f}" if stats['avg'] else "N/A",
            f"{stats['median']:.4f}" if stats['median'] else "N/A",
            f"{stats['p95']:.4f}" if stats['p95'] else "N/A", 
            f"{stats['max']:.4f}" if stats['max'] else "N/A",
            f"{stats['successful']}/{stats['total']}",
            f"{stats['success_rate']:.1f}%"
        ])
    
    print("\nResults Summary:")
    print(tabulate(
        table_data, 
        headers=["Endpoint", "Min (s)", "Avg (s)", "Median (s)", "p95 (s)", "Max (s)", "Success", "Rate"],
        tablefmt="grid"
    ))
    
    # Save report to file if requested
    if args.output:
        report = {
            "timestamp": timestamp,
            "base_url": BASE_URL,
            "iterations": args.iterations,
            "timeout": args.timeout,
            "results": all_results
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBenchmark report saved to {args.output}")
    
    # Generate chart if requested
    if args.chart:
        plot_results(all_results, args.chart)

if __name__ == "__main__":
    main()
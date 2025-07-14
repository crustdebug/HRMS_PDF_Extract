def calculate_avg_response_time(filename="response_times.txt"):
    try:
        with open(filename, "r") as f:
            times = [float(line.strip()) for line in f if line.strip()]
        if times:
            avg_time = sum(times) / len(times)
            print(f"Average response time: {avg_time:.3f} seconds over {len(times)} responses.")
        else:
            print("No response times recorded.")
    except FileNotFoundError:
        print(f"File '{filename}' not found. No response times have been logged yet.")

if __name__ == "__main__":
    calculate_avg_response_time() 
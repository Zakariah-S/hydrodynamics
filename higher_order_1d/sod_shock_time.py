import subprocess

# Define the cells values and the output file
cells_values = [200, 400, 800]
output_file = "high_order_time.txt"

# Open the output file to write the results
with open(output_file, "w") as file:
    file.write("Grid Size\tTime Taken (s)\n")
    file.write("========================\n")

    for cells in cells_values:
        # Run the sod_shock.py file with the current number of cells
        print(f"Running simulation with Grid Size: {cells}")
        result = subprocess.run(
            ["python3", "sodshock.py", str(cells)],
            capture_output=True,
            text=True
        )

        # Extract the time from the output
        output_lines = result.stdout.split("\n")
        elapsed_time = None
        for line in output_lines:
            if "Simulation completed in" in line:  # Look for the specific line
                elapsed_time = float(line.split("in")[1].split("seconds")[0].strip())
                print(f"Grid Size: {cells}, {line}")  # Output to terminal
                file.write(f"{cells}\t{elapsed_time:.2f}\n")  # Write to file
                break

        # Handle cases where the time is not found
        if elapsed_time is None:
            print(f"Grid Size: {cells}, Time Taken: Not Found (Check sod_shock.py output)")
            file.write(f"{cells}\tTime Not Found\n")

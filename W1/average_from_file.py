def average_from_file(filename):
    try :
        with open(filename, "r") as file:
            numbers = [float(line.strip()) for line in file.readlines()]
            if not numbers:
                return 0.0
            return sum(numbers) / len(numbers)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filename} not found")
    except ValueError:
        raise ValueError("File contains non-numeric data")
print (average_from_file("numbers.txt"))   
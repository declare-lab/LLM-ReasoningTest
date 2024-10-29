def split_test_cases(input_string):
    lines = input_string.split("\n")
    test_cases = []
    function = []

    for line in lines:
        if line.startswith("assert "):
            test_cases.append("\n".join(function) + "\n" + line)
        else:
            function.append(line)
    return test_cases

import random

def generate_random(value):
    while True:
        if "integer" in value["type"]:
            yield random.randint(value["minimum"], value["maximum"])
        elif "string" in value["type"]:
            if isinstance(value["enum"], list):
                yield random.choice(value["enum"])

def parsing_json_schema(json_schema):
    parsed_schema = {}
    for key, value in json_schema.items():
        parsed_schema[key] = generate_random(value)

    return parsed_schema
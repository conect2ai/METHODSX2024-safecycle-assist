# Introduction

This repository contains the code for the method "SafeCycle-Assist", designed to support safer cycling patterns in urban environments, leveraging for that Large Language Models (LLMs), AI-based agents, and open geospatial data. This repository is an implementation of the method, which employs the OpenAI API to generate a risk assessment for a cycling trip or point of interest, as well as data from the [CityZones](https://github.com/daniel-gcosta/cityzones-web) tool. To contextualize the model, the prompt in [context](context.md) was used as one of the inputs.

# Requirements

To run the code, you need to clone the repository and install the requirements:

1. Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/conect2ai/METHODXS2024-safecycle-assist.git && cd METHODSX2024-safecycle-assist
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv && source venv/bin/activate
```

3. Install the requirements:

```bash
pip install -r requirements.txt
```

# Usage

Initially, you need to set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="<INSERT API KEY HERE>"
```

To run the code, you can use the following command:

```bash
python app.py --prompt "<INSERT PROMPT HERE>"
```

# Examples

Here are some examples of how to use the code:

## Example 1

```bash
python app.py --prompt "What is the risk of a cycling trip starting from point with latitude 41.1246 and longitude -8.5717, ending at latitude 41.1570 and longitude -8.6393"
```

## Example 2

```bash
python app.py --prompt "I am planning a cycling trip that will cross the point at latitude 41.1246 and longitude -8.5717. What is the risk associate to this zone?"
```

# Results

The code will output the risk associated with the trip.

## Example 1

Output for the [Example 1](#example-1):

> The computed risk levels for the cycling trip from the starting point at latitude 41.1246 and longitude -8.5717 to the ending point at latitude 41.1570 and longitude -8.6393 are as follows:
> - Low Risk: 53.79%
> - Medium Risk: 30.30%
> - High Risk: 15.91%
>
> Based on these risk percentages, the route is considered to have a low risk overall. It is suitable for most cyclists, including beginners. The path is within close proximity to emergency services, ensuring a safer journey. Remember to stay on designated cycling lanes for maximum safety. Enjoy your ride!
    
## Example 2

Output for the [Example 2](#example-2):

>The risk level associated with the point at latitude 41.1246 and longitude -8.5717 is high (Level 3). This indicates that the area poses significant hazards and is recommended only for very experienced cyclists. It is far from emergency services and other emergency-related amenities.
>
>Given the high risk level, I recommend considering an alternative route to ensure your safety. If you provide me with the starting and ending points of your trip, I can assist you in evaluating a safer route. Remember to wear appropriate safety gear, check the weather conditions, carry a basic repair kit, stay hydrated, and share your route details with someone in case of an emergency.  
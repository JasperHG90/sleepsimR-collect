# Base image --> unix with miniconda3 distribution
FROM continuumio/miniconda3

# Install dependencies
RUN pip install --no-cache-dir daiquiri mne numpy bs4

# Copy files
RUN mkdir app
COPY frequency_bands.json frequency_bands.json
COPY pipeline.py pipeline.py
COPY utils.py utils.py

# Set up the entrypoint
ENTRYPOINT ["python", "-u", "./pipeline.py"]

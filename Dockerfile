FROM coinstac/coinstac-base-python-stream

# Set the working directory
WORKDIR /computation

# Copy the current directory contents into the container
ADD . /computation

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

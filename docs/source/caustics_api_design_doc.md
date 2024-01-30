# Caustics API Design Document

## Introduction

Welcome to the caustics API Design Document. This document outlines the
architecture, design principles, and functionalities of the Caustics Python
package's API. The API is crafted to streamline the user experience, making
caustics simulations more accessible and intuitive.

### Project Overview

The caustics API is an integral part of the larger caustics project. The API
serves as a user-friendly interface, simplifying the simulation process and
enabling users to customize simulations based on parameters such as lens type,
light source characteristics, and forward function routines.

### Objectives

The primary objectives of the caustics API design are as follows:

1. **Simplicity and Usability:** Design an API that is easy to understand,
   straightforward to use, and enhances the overall user experience.

2. **Flexibility and Modularity:** Create a modular architecture that allows
   users to adapt the simulation environment to their specific needs, promoting
   code reusability and extensibility.

3. **Efficiency and Performance:** Leverage caustic's native GPU acceleration
   and automatic differentiation to optimize the simulation process without
   needing to know low-level functionality, providing users with fast and
   accurate results.

### Document Structure

This design document is organized into sections that cover various aspects of
the API, including:

- **Overview:** A high-level description of the caustics project and the role of
  the API.
- **Functionality:** Details on the three main functions of the API—creating a
  simulator, running the forward function routine, and plotting results.
- **Usage Examples:** Practical examples demonstrating how to use the API for
  common simulation scenarios.
- **Testing and Validation:** Strategies for testing the API's functionality and
  ensuring its reliability.

## Overview

The caustics project represents a pioneering lensing simulation toolkit designed
to push the boundaries of realism and versatility in caustics generation. At its
core, caustics leverages advanced computational techniques, including GPU
acceleration and automatic differentiation, to simulate intricate light patterns
formed through the reflection or refraction of light. These simulations find
applications in various fields, from computer graphics and physics to optics and
beyond. The caustics API plays a pivotal role within this project, serving as
the gateway for users to harness the power of the underlying simulation engine.
By encapsulating the intricacies of the simulation process, the API provides
users with a streamlined and user-friendly interface. Its three main
functions—creating a simulator, running the forward function routine, and
plotting results—facilitate a seamless and customizable experience, empowering
users to effortlessly conduct caustics simulations tailored to their specific
needs. Through the Caustics API, the project endeavors to democratize access to
advanced lensing simulations, making it a valuable tool for researchers,
developers, and enthusiasts alike.

## Functionality

### Simulator

- Builds the simulator based on user-defined parameters

### Forward

- Runs a set of pre-defined and/or user-defined forward routines

### Plot

- This section of the API will be finalized at a later time.

## Usage Examples

### Simulator

```python
sim = caustics.build_simulator(input)  # input can be pydantic model/yaml file path
```

#### Input Parameters

Template yaml

```yaml
simulator:
  name: "simulator_name"
  lens:
    name: "sie"
    kind: "SIE"
    cosmology:
      name: "cosmo"
      kind: "FlatLambdaCDM"
      params: {}
    params: {}
    from_file: # optional
  source:
    name: "sersic"
    kind: "Sersic"
    params: {}
    from_file: # optional
  params:
    z_s: 1.0
  forward:
    routine: "default_routine?"
    from_file: # optional
  state: # optional
    load: # optional
    save: # optional
```

Multiple lenses example

```yaml
simulator:
    name: "simulator_name"
    lens:
        name:
        kind: "Multiplane"
        lenses:
        - name: "lens1"
          kind: "SIE"
          params: {}
          from_file: # optional
        - name: "lens2"
          kind: "SIE"
          params: {}
          from_file: # optional
        ...
    ... # same as above
```

#### Output

- Temporary YAML template file for user to fill in. This file will be stored in
  a current working directory for user to be able to access.

Example YAML file

```yaml
params:
  z_l: # What is it (Suggested Units: )
  lens_name:
    x0: 0.7 # What is it (Suggested Units: )
    y0: 0.13 # What is it (Suggested Units: )
    q: 0.4 # What is it (Suggested Units: )
    phi: np.pi / 5 # What is it (Suggested Units: )
    b: 1.0 # What is it (Suggested Units: )
  source_name:
    x0: 0.2 # What is it (Suggested Units: )
    y0: 0.5 # What is it (Suggested Units: )
    q: 0.5 # What is it (Suggested Units: )
    phi: -np.pi / 4 # What is it (Suggested Units: )
    n: 1.5 # What is it (Suggested Units: )
    Re: 2.5 # What is it (Suggested Units: )
    Ie: 1.0 # What is it (Suggested Units: )
```

### Forward

```python
results = sim(input)
```

#### Input Parameters

- Accepts loaded `state_dict`, temporary `yaml`, or direct variable input

#### Output

- Routine calculations

### Plot

```
TBD
```

#### Input Parameters

- TBD

#### Output

- Data plot(s)

## Testing and Validation

Pydantic is a Python library for data validation and settings management, and
it's commonly used for API validation. Here's an explanation of how Pydantic
will be used for API validation:

1. **Defining Data Models:**

   - In Pydantic, you create data models using Python classes. Each attribute of
     the class represents a field in the data model.
   - You can specify the data type of each field, as well as additional
     constraints such as minimum and maximum values, and even create nested
     models for more complex structures.

   ```python
   from pydantic import BaseModel


   class User(BaseModel):
       username: str
       email: str
       age: int
   ```

2. **Request Data Validation:**

   - When handling incoming API requests, you can use Pydantic models to
     validate the request data against the defined data model.
   - Pydantic will automatically validate the data types and constraints, and it
     can also provide meaningful error messages if validation fails.

   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel

   app = FastAPI()


   @app.post("/create_user/")
   async def create_user(user: User):
       # If the request data doesn't match the User model, FastAPI will automatically
       # respond with a 422 Unprocessable Entity status and detailed error information.
       return {"username": user.username, "email": user.email, "age": user.age}
   ```

3. **Response Data Validation:**

   - Pydantic can also be used to validate and serialize the response data
     before sending it back to the client.
   - You can define response models in a similar way to request models, ensuring
     that the data returned adheres to a specific structure.

   ```python
   class UserResponse(BaseModel):
       username: str
       email: str


   @app.post("/create_user/", response_model=UserResponse)
   async def create_user(user: User):
       pass
       # ... process the request and return a response that matches the UserResponse model
   ```

4. **Automatic Documentation:**
   - When using Pydantic models, automatic API documentation is generated.
   - This documentation includes details about the expected request and response
     structures based on the Pydantic models.

By leveraging Pydantic for API validation, we will enhance the robustness of the
API by ensuring that incoming data adheres to the expected structure and
constraints. This not only helps catch potential errors early in the process but
also provides automatic documentation to facilitate communication between
researchers and contributors using the API.

## Action Plan

1. Create Pydantic models
   1. Common
   2. Simulator
   3. Simulator input
   4. MultiLens
2. Create validations
3. Create functions
4. Create tests
5. Create documentation

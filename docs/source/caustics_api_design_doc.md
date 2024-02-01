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
functions-validating parameter inputs, creating a simulator, and running 
the forward function routine—facilitate a seamless and customizable experience, empowering
users to effortlessly conduct caustics simulations tailored to their specific
needs. Through the Caustics API, the project endeavors to democratize access to
advanced lensing simulations, making it a valuable tool for researchers,
developers, and enthusiasts alike.

## Functionality/Usage Examples

### Input YAML File

Define pre-validated `build_simulator` parameters.

```yaml
simulator:
    name: "sim"
    kind: "Simulator"
    params:
        z_s: 0.8
    kwargs:
        pixelscale: 0.3
        pixels_x: 1
        lens_light: true
        psf: 3.1
        pixels_y: 100
        upsample_factor: 1
        psf_pad: true
        psf_mode: "fft"
    lens:
        name: "sie"
        kind: "SIE"
        params: {}
        cosmology:
            name: "cosmo"
            kind: "FlatLambdaCDM"
            params:
                Om0: 0.3
                critical_density0: 127052816384
                h0: 0.67
    src:
        name: "src"
        kind: "Sersic"
        params: {}
    forward: |
    def forward(self, params):
        # Here the simulator unpacks the parameter it needs
        z_s = self.unpack(params)

        # Note this is very similar to before, except the packed up `x` is all the raytrace function needs to work
        bx, by = self.lens.raytrace(thx, thy, z_s, params)
        mu_fine = self.src.brightness(bx, by, params)

        # We return the sampled brightness at each pixel location
        return avg_pool2d(mu_fine.squeeze()[None, None], upsample_factor)[0, 0]
        
```

### Create Pydantic Models

Define Pydantic models to validate the input YAML structure. We need models for the overall simulator, lens, source, and cosmology. Here's a simplified example:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Cosmology(BaseModel):
    name: str
    kind: str
    params: dict

class Lens(BaseModel):
    name: str
    kind: str
    params: dict
    cosmology: Optional[Cosmology]

class Source(BaseModel):
    name: str
    kind: str
    params: dict

class SimulatorInput(BaseModel):
    name: str
    kind: str
    params: dict
    kwargs: dict
    lens: Lens
    src: Source
    forward: str
```

#### Create a Registry

Maintain a registry of simulator classes, lenses, and sources. We will use a dictionary for this purpose.

### Build Simulators Dynamically

Create a function to build simulators dynamically based on the validated input YAML. We will use Python's `type` function to dynamically create classes. Here's a simplified example:


#### Part 1: Build Simulator Class

```python
sim = caustics.build_simulator(input)  # input can be pydantic model/yaml file path
```

Simplified simulator class example

```python
from typing import Type, Dict

def build_simulator(input_yaml: dict) -> Type:
    class_params = input_yaml['simulator']['params']
    class_name = input_yaml['simulator']['name']
    
    class CustomSimulator(BaseModel):
        class Config:
            allow_mutation = False
        ...
        lens: Lens = Lens(**input_yaml['lens'])
        src: Source = Source(**input_yaml['src'])
        forward: str = input_yaml['simulator']['forward']

    return CustomSimulator
```

### Part 2: Hook Forward Implementation

Since the forward can contain custom functions and combination of functions, the forward will be directly accepted as Python code (except for "Lens_Source" because it contains a pre-defined forward).
The forward will use the `@hookimp` for allowing abstraction and real-time implementation.

### Part 3: Output Input Signature

Outputs necessary instructions about required variables along with order and other relevant information for input into the simulator for completing the task.

```python
sim.input_signature()
```

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

def calculate_demand_trip(
    population: int,
    car_ownership_rate: float,
    public_transport_usage_rate: float,
    average_trips_per_person: int
) -> dict:
    """
    Calculate the demand for different transportation modes based on input parameters.
    
    Args:
        population (int): Total population of the city.
        car_ownership_rate (float): Percentage of population that owns a car (0-1).
        public_transport_usage_rate (float): Percentage of population that uses public transport (0-1).
        average_trips_per_person (int): Average number of trips made by each person per day.
    
    Returns:
        dict: A dictionary containing estimated demand for cars, public transport, and other modes.
    """
    
    # Calculate total trips in the city
    total_trips = population * average_trips_per_person
    
    # Calculate demand for cars
    car_demand = total_trips * car_ownership_rate
    
    # Calculate demand for public transport
    public_transport_demand = total_trips * public_transport_usage_rate
    
    # Calculate demand for other modes (walking, biking, etc.)
    other_modes_demand = total_trips - (car_demand + public_transport_demand)
    
    return {
        "total_trips": total_trips,
        "car_demand": car_demand,
        "public_transport_demand": public_transport_demand,
        "other_modes_demand": other_modes_demand
    }
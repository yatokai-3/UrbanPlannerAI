
''' How many people actually using this service or will use'''
def calculate_transit_ridership(
    route_length_km: float,
    number_of_stops: int,
    mode_type: str,
    population_within_500m: int,
    trip_rate:float|2.5, #by default 2.5 trips per person per day
    capture_rate_bus=0.06,
    capture_rate_metro=0.20,
    capture_rate_brt=0.12,


) -> dict:
    """
    Calculate the demand for different transportation modes based on input parameters.
    
    Args:
        population_within_500m (int): Population within a 500m radius of the route.
        trip_rate (float): Rate of trips per person per day.
    
    Returns:
        dict: A dictionary containing estimated The calculator provides:
        Average Stop Spacing
        Coverage Factor
        Estimated Daily Ridership
        Estimated Peak-Hour Ridership
    """
    
    #average stop spacing
    average_stop_spacing = route_length_km / number_of_stops
    
    #coverage factor
    coverage_factor = number_of_stops / (route_length_km / average_stop_spacing)
    
    # Calculate total trips in the city
    total_trips = population_within_500m * trip_rate
    
    
    # Calculate demand for other modes (walking, biking, etc.)
    other_modes_demand = total_trips - (car_demand + public_transport_demand)
    
    return {
        "total_trips": total_trips,

    }
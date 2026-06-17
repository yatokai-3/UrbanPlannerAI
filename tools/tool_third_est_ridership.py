
''' How many people actually using this service or will use'''
def calculate_transit_ridership(
    route_length_km: float,
    number_of_stops: int,
    mode_type: str,
    population_within_500m: int,
    trip_rate=2.5, #by default 2.5 trips per person per day

    ##capture rates for different modes of transport
    capture_rate_bus=0.06,
    capture_rate_metro=0.20,
    capture_rate_brt=0.12,
    capture_rate_lrt=0.15,
    

    ## car competition factors for different modes of transport
    ccf_bus=2.0,
    ccf_BRT=1.5,
    ccf_metro=1.0,
    ccf_LRT=1.3


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
    catchment_per_stop=1.0 #0.5km x 2 (both sides)
    total_catchment_area = catchment_per_stop * number_of_stops
    coverage_factor = round(total_catchment_area / route_length_km,2)

    ## total corridor trips
    total_trips = population_within_500m * trip_rate

    ## capture rate based on mode type
    if mode_type.lower() == 'bus':
        capture_rate = capture_rate_bus
        ccf = ccf_bus
    elif mode_type.lower() == 'metro':
        capture_rate = capture_rate_metro
        ccf = ccf_metro
    elif mode_type.lower() == 'brt':
        capture_rate = capture_rate_brt
        ccf = ccf_BRT
    elif mode_type.lower() == 'lrt':
        capture_rate = capture_rate_lrt  # Assuming similar capture rate for BRT and LRT
        ccf = ccf_LRT
    else:
        raise ValueError("Invalid mode type. Choose from 'bus', 'metro', 'brt', or 'lrt'.")
    

    ## apply coverage factor and capture rate to estimate ridership
    ## rider within reach of the stops and who would choose this mode over others
    trip_on_this_mode = total_trips * coverage_factor * capture_rate

    ## adjust for car competition factor
    adjusted_daily_ridership = round(trip_on_this_mode / ccf,3)

    ## Peak hour
    peak_hour_ridership = round(adjusted_daily_ridership * 0.12,3)  # Assuming 12% of daily ridership occurs during peak hours
    
    
    return{
        "stopping_spacing_km": str(average_stop_spacing) + " km/stop",
        "coverage_factor": str(coverage_factor * 100) + " % of corridor covered", 
        "trip_on_this_mode": str(trip_on_this_mode) + " daily trips",
        "estimated/adjusted_daily_ridership": str(adjusted_daily_ridership) + " daily trips",
        "estimated_peak_hour_ridership": str(peak_hour_ridership) + " peak hour trips"
    }


print(calculate_transit_ridership(
    route_length_km=48,
    number_of_stops=37,
    mode_type='metro',
    population_within_500m=200000
))
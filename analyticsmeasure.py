"""
Amplitude Skin Analytics Tracker
Sends skin analysis data to Amplitude for tracking over time
"""

from amplitude import Amplitude, BaseEvent
from datetime import datetime
import json

# -----------------------------
# Amplitude Configuration
# -----------------------------
AMPLITUDE_API_KEY = "---"

# Initialize Amplitude client
amplitude_client = Amplitude(AMPLITUDE_API_KEY)


class SkinAnalyticsTracker:
    """
    Track skin analysis data in Amplitude
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the tracker
        
        Parameters:
        - api_key:  Your Amplitude API key (optional, uses default if not provided)
        """
        if api_key: 
            self.client = Amplitude(api_key)
        else:
            self. client = amplitude_client
    
    
    def track_analysis(self, user_id, acne_data, skin_data=None, metadata=None):
        """
        Track a skin analysis event in Amplitude
        
        Parameters: 
        - user_id:  Unique identifier for the user (email, username, UUID, etc.)
        - acne_data: Dictionary with acne detection results
            {
                'has_acne': bool,
                'spot_count': int,
                'severity': str,  # "None", "Minimal", "Mild", "Moderate", "Severe"
                'confidence': str,  # "High", "Medium", "Low"
                'acne_locations': list  # Optional:  list of spot locations
            }
        - skin_data: Dictionary with skin quality metrics (optional)
            {
                'oiliness': str,  # "Dry", "Normal", "Oily"
                'redness_level': str,  # "Low", "High"
                'brightness': float,  # 0-255
                'redness':  float,  # LAB 'a' channel value
                'texture': float,  # Variance value
                'left_cheek':  dict,  # Optional regional metrics
                'right_cheek':  dict,  # Optional
                'forehead': dict  # Optional
            }
        - metadata: Additional custom properties (optional)
            {
                'image_source': str,  # "webcam", "upload", etc.
                'lighting_condition': str,
                'time_of_day':  str,
                etc.
            }
        
        Returns:
        - bool: True if successful, False otherwise
        """
        
        try:
            # Validate required acne data
            if not isinstance(acne_data, dict):
                raise ValueError("acne_data must be a dictionary")
            
            required_fields = ['has_acne', 'spot_count', 'severity', 'confidence']
            for field in required_fields:
                if field not in acne_data:
                    raise ValueError(f"acne_data missing required field: {field}")
            
            # Build event properties
            event_properties = {
                # Acne metrics
                'has_acne': bool(acne_data['has_acne']),
                'acne_spot_count': int(acne_data['spot_count']),
                'acne_severity': str(acne_data['severity']),
                'confidence': str(acne_data['confidence']),
                
                # Timestamp
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'analysis_time': datetime.now().strftime('%H:%M:%S'),
            }
            
            # Add acne location data if available
            if 'acne_locations' in acne_data and acne_data['acne_locations']:
                locations = acne_data['acne_locations']
                event_properties['acne_location_count'] = len(locations)
                
                # Calculate average spot size
                if locations:
                    avg_area = sum(loc. get('area', 0) for loc in locations) / len(locations)
                    event_properties['avg_spot_size'] = float(avg_area)
            
            # Add skin quality metrics if available
            if skin_data and isinstance(skin_data, dict):
                event_properties.update({
                    'skin_oiliness': str(skin_data.get('oiliness', 'Unknown')),
                    'skin_redness_level': str(skin_data.get('redness_level', 'Unknown')),
                    'skin_brightness': float(skin_data.get('brightness', 0)),
                    'skin_redness_value': float(skin_data.get('redness', 0)),
                    'skin_texture': float(skin_data.get('texture', 0)),
                })
                
                # Add regional metrics if available
                if 'left_cheek' in skin_data and skin_data['left_cheek']:
                    event_properties['left_cheek_brightness'] = float(skin_data['left_cheek'].get('brightness', 0))
                    event_properties['left_cheek_redness'] = float(skin_data['left_cheek'].get('redness', 0))
                    event_properties['left_cheek_texture'] = float(skin_data['left_cheek'].get('texture', 0))
                
                if 'right_cheek' in skin_data and skin_data['right_cheek']: 
                    event_properties['right_cheek_brightness'] = float(skin_data['right_cheek'].get('brightness', 0))
                    event_properties['right_cheek_redness'] = float(skin_data['right_cheek'].get('redness', 0))
                    event_properties['right_cheek_texture'] = float(skin_data['right_cheek'].get('texture', 0))
                
                if 'forehead' in skin_data and skin_data['forehead']:
                    event_properties['forehead_brightness'] = float(skin_data['forehead'].get('brightness', 0))
                    event_properties['forehead_redness'] = float(skin_data['forehead'].get('redness', 0))
                    event_properties['forehead_texture'] = float(skin_data['forehead'].get('texture', 0))
            
            # Add custom metadata if provided
            if metadata and isinstance(metadata, dict):
                for key, value in metadata.items():
                    event_properties[key] = value
            
            # Create and send event
            event = BaseEvent(
                event_type="Skin Analysis Completed",
                user_id=str(user_id),
                event_properties=event_properties
            )
            
            self.client.track(event)
            self.client.flush()
            
            print("\n✓ Skin analysis data sent to Amplitude successfully!")
            print(f"  User ID: {user_id}")
            print(f"  Acne Severity: {acne_data['severity']}")
            print(f"  Spot Count: {acne_data['spot_count']}")
            if skin_data:
                print(f"  Skin Oiliness: {skin_data.get('oiliness', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error sending data to Amplitude: {e}")
            return False
    
    
    def track_user_profile(self, user_id, user_properties):
        """
        Set user properties in Amplitude (demographic data, skin type, etc.)
        
        Parameters:
        - user_id:  Unique identifier for the user
        - user_properties: Dictionary of user attributes
            {
                'age': int,
                'gender':  str,
                'skin_type': str,  # "Oily", "Dry", "Combination", "Normal"
                'primary_concern': str,  # "Acne", "Redness", "Texture", etc.
                'signup_date': str,
                etc.
            }
        
        Returns:
        - bool:  True if successful, False otherwise
        """
        
        try:
            # Create identify event to set user properties
            identify_event = BaseEvent(
                event_type="$identify",
                user_id=str(user_id),
                user_properties=user_properties
            )
            
            self. client.track(identify_event)
            self.client.flush()
            
            print(f"\n✓ User profile updated in Amplitude for user: {user_id}")
            return True
            
        except Exception as e:
            print(f"\n✗ Error updating user profile: {e}")
            return False
    
    
    def track_improvement(self, user_id, previous_analysis, current_analysis):
        """
        Track improvement/change between two analyses
        
        Parameters: 
        - user_id:  Unique identifier for the user
        - previous_analysis: Previous acne_data dictionary
        - current_analysis: Current acne_data dictionary
        
        Returns:
        - bool: True if successful, False otherwise
        """
        
        try:
            # Calculate changes
            spot_change = current_analysis['spot_count'] - previous_analysis['spot_count']
            spot_change_percent = (spot_change / previous_analysis['spot_count'] * 100) if previous_analysis['spot_count'] > 0 else 0
            
            severity_map = {"None": 0, "Minimal": 1, "Mild":  2, "Moderate": 3, "Severe": 4}
            prev_severity_level = severity_map.get(previous_analysis['severity'], 0)
            curr_severity_level = severity_map.get(current_analysis['severity'], 0)
            severity_change = curr_severity_level - prev_severity_level
            
            # Determine if improved, worsened, or stayed same
            if spot_change < 0:
                change_direction = "Improved"
            elif spot_change > 0:
                change_direction = "Worsened"
            else:
                change_direction = "No Change"
            
            # Create improvement event
            event_properties = {
                'previous_spot_count': previous_analysis['spot_count'],
                'current_spot_count': current_analysis['spot_count'],
                'spot_change': spot_change,
                'spot_change_percent': round(spot_change_percent, 2),
                
                'previous_severity': previous_analysis['severity'],
                'current_severity': current_analysis['severity'],
                'severity_change':  severity_change,
                
                'change_direction': change_direction,
                'timestamp': datetime.now().isoformat(),
            }
            
            event = BaseEvent(
                event_type="Skin Analysis Improvement Tracked",
                user_id=str(user_id),
                event_properties=event_properties
            )
            
            self.client.track(event)
            self.client.flush()
            
            print(f"\n✓ Improvement tracking sent to Amplitude!")
            print(f"  Change: {change_direction}")
            print(f"  Spot Count:  {previous_analysis['spot_count']} → {current_analysis['spot_count']} ({spot_change: +d})")
            print(f"  Severity: {previous_analysis['severity']} → {current_analysis['severity']}")
            
            return True
            
        except Exception as e: 
            print(f"\n✗ Error tracking improvement: {e}")
            return False


# -----------------------------
# Convenience functions
# -----------------------------

def track_skin_analysis(user_id, acne_data, skin_data=None, metadata=None, api_key=None):
    """
    Quick function to track a skin analysis
    
    Parameters:  Same as SkinAnalyticsTracker. track_analysis()
    
    Returns: 
    - bool: True if successful, False otherwise
    """
    tracker = SkinAnalyticsTracker(api_key=api_key)
    return tracker.track_analysis(user_id, acne_data, skin_data, metadata)


def set_user_profile(user_id, user_properties, api_key=None):
    """
    Quick function to set user properties
    
    Parameters: Same as SkinAnalyticsTracker. track_user_profile()
    
    Returns:
    - bool: True if successful, False otherwise
    """
    tracker = SkinAnalyticsTracker(api_key=api_key)
    return tracker.track_user_profile(user_id, user_properties)


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AMPLITUDE SKIN ANALYTICS TRACKER - EXAMPLE USAGE")
    print("="*70)
    
    # Example 1: Basic acne tracking
    print("\n--- Example 1: Basic Acne Tracking ---")
    
    acne_data = {
        'has_acne': True,
        'spot_count': 12,
        'severity': 'Mild',
        'confidence': 'High',
        'acne_locations':  [
            {'center': (100, 150), 'radius': 5, 'area': 78.5},
            {'center': (200, 180), 'radius': 6, 'area': 113.1},
        ]
    }
    
    track_skin_analysis(
        user_id="user@example.com",
        acne_data=acne_data
    )
    
    # Example 2: Full tracking with skin metrics
    print("\n--- Example 2: Full Tracking with Skin Metrics ---")
    
    acne_data = {
        'has_acne': True,
        'spot_count': 8,
        'severity': 'Mild',
        'confidence': 'High'
    }
    
    skin_data = {
        'oiliness': 'Normal',
        'redness_level': 'Low',
        'brightness': 145.3,
        'redness': 128.7,
        'texture':  245.9,
        'left_cheek': {
            'brightness': 142.1,
            'redness': 130.2,
            'texture': 250.1
        },
        'right_cheek': {
            'brightness': 148.5,
            'redness': 127.2,
            'texture': 241.7
        },
        'forehead': {
            'brightness': 150.0,
            'redness': 125.0,
            'texture': 240.0
        }
    }
    
    metadata = {
        'image_source': 'webcam',
        'lighting_condition': 'indoor',
        'time_of_day': 'morning'
    }
    
    track_skin_analysis(
        user_id="user@example.com",
        acne_data=acne_data,
        skin_data=skin_data,
        metadata=metadata
    )
    
    # Example 3: Set user profile
    print("\n--- Example 3: Set User Profile ---")
    
    user_properties = {
        'age': 25,
        'gender': 'Female',
        'skin_type': 'Combination',
        'primary_concern':  'Acne',
        'signup_date': '2024-01-15'
    }
    
    set_user_profile(
        user_id="user@example.com",
        user_properties=user_properties
    )
    
    # Example 4: Track improvement over time
    print("\n--- Example 4: Track Improvement ---")
    
    tracker = SkinAnalyticsTracker()
    
    previous_analysis = {
        'has_acne': True,
        'spot_count': 15,
        'severity': 'Moderate',
        'confidence': 'High'
    }
    
    current_analysis = {
        'has_acne': True,
        'spot_count': 8,
        'severity': 'Mild',
        'confidence':  'High'
    }
    
    tracker.track_improvement(
        user_id="user@example.com",
        previous_analysis=previous_analysis,
        current_analysis=current_analysis
    )
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
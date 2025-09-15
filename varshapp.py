from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from dash_extensions import EventListener
import pandas as pd
import numpy as np
import json
import os
from dynamic import PerUserKeystrokeAuth  # Import your actual class

# --- Load available users from trained models ---
def get_available_users():
    models_dir = "user_models"
    if not os.path.exists(models_dir):
        return []
    
    user_files = [f for f in os.listdir(models_dir) if f.startswith("model_") and f.endswith(".h5")]
    available_users = sorted([f[len("model_"):-len(".h5")] for f in user_files])
    return available_users

available_users = get_available_users()

# --- Load authentication system ---
auth_system = PerUserKeystrokeAuth(data_path="/Users/venugopal/Downloads/free-text.csv")
auth_system.load_models()

# --- App setup ---
app = Dash(__name__)
app.title = "Real-Time Keystroke Authentication"

# Event configuration for keystroke capture
event_config = [
    {
        "event": "keydown",
        "props": ["key", "timeStamp", "type"]
    },
    {
        "event": "keyup", 
        "props": ["key", "timeStamp", "type"]
    }
]

app.layout = html.Div([
    html.H2("🔐 Real-Time Keystroke Authentication"),
    html.P("Type below. Keystroke timing data will be used to authenticate using trained models."),

    html.Label("Select user:"),
    dcc.Dropdown(
        id='user-dropdown',
        options=[{'label': user, 'value': user} for user in available_users],
        value=available_users[0] if available_users else None,
        style={'width': '300px', 'marginBottom': '20px'}
    ),

    html.Label("Typing area:"),
    html.Div([
        EventListener(
            id="keystroke-listener",
            events=event_config,
            children=dcc.Textarea(
                id='typed-text',
                rows=4,
                style={'width': '400px', 'marginTop': '10px'},
                placeholder='Start typing here...',
                value=''
            )
        )
    ]),

    # Data stores
    dcc.Store(id='keystroke-store', data=[]),
    dcc.Store(id='processed-features', data=None),
    
    html.Br(),
    html.Button("Authenticate", id="auth-button", n_clicks=0),
    html.Button("Reset", id="reset-button", n_clicks=0, style={"marginLeft": "10px"}),
    html.Button("Show Debug", id="debug-button", n_clicks=0, style={"marginLeft": "10px"}),

    html.Br(), html.Br(),
    html.Div(id='auth-output', style={'fontSize': 20, 'fontWeight': 'bold'}),
    html.Div(id='debug-output', style={'marginTop': '20px', 'padding': '10px', 
                                      'backgroundColor': '#f0f0f0', 'display': 'none'})
])

# --- Callback 1: Capture keystroke events ---
@app.callback(
    Output('keystroke-store', 'data'),
    Input('keystroke-listener', 'event'),
    State('keystroke-store', 'data'),
    prevent_initial_call=True
)
def capture_keystrokes(event, current_data):
    """Capture keystroke events"""
    if event is None:
        return no_update
    
    # Process the event
    event_data = event.copy()
    event_data['timestamp'] = event_data.pop('timeStamp', 0)
    
    return current_data + [event_data]

# --- Callback 2: Process features for authentication ---
@app.callback(
    Output('processed-features', 'data'),
    Input('keystroke-store', 'data'),
    prevent_initial_call=True
)
def process_keystroke_features(keystrokes):
    """Process raw keystrokes into timing features compatible with trained models"""
    if not keystrokes or len(keystrokes) < 4:
        return None
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(keystrokes)
        
        # Filter valid keys (single characters only)
        if 'key' in df.columns:
            df = df[df['key'].str.len() == 1]
        else:
            return None
        
        if len(df) < 4:
            return None

        # Separate keydown and keyup events
        down_df = df[df['type'] == 'keydown'].copy()
        up_df = df[df['type'] == 'keyup'].copy()
        
        if len(down_df) == 0 or len(up_df) == 0:
            return None

        # Calculate timing features to match trained model format
        timing_features = []
        
        # Sort by timestamp to get proper sequence
        down_df = down_df.sort_values('timestamp')
        up_df = up_df.sort_values('timestamp')
        
        # Calculate dwell times (key hold times) and flight times
        prev_down_time = None
        prev_up_time = None
        
        for i, down_row in down_df.iterrows():
            key = down_row['key']
            down_time = down_row['timestamp']
            
            # Find corresponding keyup event
            matching_ups = up_df[(up_df['key'] == key) & (up_df['timestamp'] > down_time)]
            if len(matching_ups) > 0:
                up_time = matching_ups.iloc[0]['timestamp']
                
                # Calculate dwell time (DU - Down to Up)
                dwell_time = (up_time - down_time) / 1000.0  # Convert to seconds
                
                # Calculate flight times if we have previous events
                dd_time = 0  # Down-Down time
                du_time = 0  # Down-Up time  
                ud_time = 0  # Up-Down time
                uu_time = 0  # Up-Up time
                
                if prev_down_time is not None:
                    dd_time = (down_time - prev_down_time) / 1000.0
                    du_time = (up_time - prev_down_time) / 1000.0
                
                if prev_up_time is not None:
                    ud_time = (down_time - prev_up_time) / 1000.0
                    uu_time = (up_time - prev_up_time) / 1000.0
                
                # Create feature vector matching the trained model format
                # The trained model expects: ['DU.key1.key1', 'DD.key1.key2', 'DU.key1.key2', 'UD.key1.key2', 'UU.key1.key2 ']
                feature_row = [
                    dwell_time,  # DU.key1.key1
                    dd_time,     # DD.key1.key2  
                    du_time,     # DU.key1.key2
                    ud_time,     # UD.key1.key2
                    uu_time      # UU.key1.key2
                ]
                
                timing_features.append(feature_row)
                
                prev_down_time = down_time
                prev_up_time = up_time
        
        if len(timing_features) < 2:
            return None
            
        return np.array(timing_features).tolist()
        
    except Exception as e:
        print(f"Error processing features: {e}")
        return None

# --- Callback 3: Authentication ---
@app.callback(
    Output('auth-output', 'children'),
    Input('auth-button', 'n_clicks'),
    [State('processed-features', 'data'), State('typed-text', 'value'), 
     State('user-dropdown', 'value')],
    prevent_initial_call=True
)
def authenticate_user(n_clicks, processed_features, typed_text, selected_user):
    """Authenticate user using trained model"""
    if not processed_features or not typed_text or len(typed_text.strip()) < 5:
        return "⌨️ Please type at least 5 characters..."

    if not selected_user:
        return "⚠️ Please select a user..."

    if selected_user not in auth_system.user_models:
        return f"⚠️ No trained model found for user {selected_user}"

    try:
        # Convert features to numpy array
        test_data = np.array(processed_features)
        
        if len(test_data) < auth_system.sequence_length:
            return f"⚠️ Need at least {auth_system.sequence_length} keystrokes. Got {len(test_data)}."

        # Use the trained model to authenticate
        result = auth_system.authenticate_user(selected_user, test_data)
        
        if result is None:
            return "⚠️ Authentication failed - unable to process data"
        
        if result['authenticated']:
            return html.Div([
                html.Span("✅ Access Granted", style={'color': 'green', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small(f"Confidence: {result['confidence']:.3f} | Sequences: {result['num_sequences']}")
            ])
        else:
            return html.Div([
                html.Span("❌ Access Denied", style={'color': 'red', 'fontWeight': 'bold'}),
                html.Br(),
                html.Small(f"Confidence: {result['confidence']:.3f} | Sequences: {result['num_sequences']}")
            ])

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return html.Div([
            html.P(f"⚠️ Authentication Error: {str(e)}", style={'color': 'red'}),
            html.Details([
                html.Summary("Error Details"),
                html.Pre(error_details, style={'fontSize': '12px'})
            ])
        ])

# --- Callback 4: Debug display ---
@app.callback(
    [Output('debug-output', 'children'), Output('debug-output', 'style')],
    Input('debug-button', 'n_clicks'),
    [State('keystroke-store', 'data'), State('processed-features', 'data')],
    prevent_initial_call=True
)
def toggle_debug(n_clicks, keystrokes, features):
    """Toggle debug information display"""
    if n_clicks % 2 == 1:  # Show debug
        debug_info = []
        
        debug_info.append(f"Raw Events Captured: {len(keystrokes) if keystrokes else 0}")
        
        if keystrokes:
            debug_info.append("\nLast 10 Raw Events:")
            for i, event in enumerate(keystrokes[-10:]):
                debug_info.append(f"  {i}: {event.get('type', 'N/A')} - {event.get('key', 'N/A')} - {event.get('timestamp', 'N/A')}")
        
        if features:
            debug_info.append(f"\nProcessed Features: {len(features)} timing vectors")
            debug_info.append("Feature format: [DU, DD, DU, UD, UU]")
            debug_info.append("\nLast 5 Feature Vectors:")
            for i, feat in enumerate(features[-5:]):
                debug_info.append(f"  {i}: {[f'{x:.4f}' for x in feat]}")
        
        debug_text = "\n".join(debug_info)
        
        return html.Pre(debug_text), {'marginTop': '20px', 'padding': '10px', 
                                     'backgroundColor': '#f0f0f0', 'display': 'block'}
    else:  # Hide debug
        return "", {'display': 'none'}

# --- Callback 5: Reset functionality ---
@app.callback(
    [Output('typed-text', 'value'), Output('keystroke-store', 'data', allow_duplicate=True)],
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_data(n_clicks):
    """Reset the typing area and keystroke data"""
    return "", []

# --- Run app ---
if __name__ == '__main__':
    if not available_users:
        print("Warning: No trained models found in 'user_models' directory!")
        print("Please run the training script first to create user models.")
    else:
        print(f"Loaded models for users: {available_users}")
    
    app.run(debug=True, port=8050)
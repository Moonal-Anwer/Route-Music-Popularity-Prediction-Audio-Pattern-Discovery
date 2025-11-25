import streamlit as st
import pandas as pd
import numpy as np
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def get_features(artist_name,track_name):
    auth_manager= SpotifyClientCredentials(
        client_id='client_id',
        client_secret='client_secret'
    )

    sp = spotipy.Spotify(auth_manager=auth_manager)


def redefining_raw_data(raw_data,scaler,encoder):
    

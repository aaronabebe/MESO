import plotly.graph_objects as go

SAILING_CLASS_DISTRIBUTION = {
    'ALGAE': 1, 'BIRD': 65, 'BOAT': 262, 'BOAT_WITHOUT_SAILS': 456, 'BUOY': 319, 'CONSTRUCTION': 207, 'CONTAINER': 51,
    'CONTAINER_SHIP': 267, 'CRUISE_SHIP': 108, 'DOLPHIN': 2, 'FAR_AWAY_OBJECT': 4650, 'FISHING_BUOY': 90,
    'FISHING_SHIP': 17, 'FLOTSAM': 261, 'HARBOUR_BUOY': 94, 'HORIZON': 1, 'HUMAN': 9, 'HUMAN_IN_WATER': 11,
    'HUMAN_ON_BOARD': 173, 'KAYAK': 3, 'LEISURE_VEHICLE': 23, 'MARITIME_VEHICLE': 936, 'MOTORBOAT': 408,
    'OBJECT_REFLECTION': 30, 'SAILING_BOAT': 534, 'SAILING_BOAT_WITH_CLOSED_SAILS': 576,
    'SAILING_BOAT_WITH_OPEN_SAILS': 528, 'SEAGULL': 3, 'SHIP': 347, 'SUN_REFLECTION': 11, 'UNKNOWN': 5,
    'WATERTRACK': 105
}


def plot_class_distribution_sailing_dataset():
    px_dict = {'class': SAILING_CLASS_DISTRIBUTION.keys(), 'count': SAILING_CLASS_DISTRIBUTION.values()}

    colors = ['lightslategray', ] * len(px_dict['class'])
    colors[10] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=list(SAILING_CLASS_DISTRIBUTION.keys()),
        y=list(SAILING_CLASS_DISTRIBUTION.values()),
        marker_color=colors
    )])

    fig.show()


if __name__ == '__main__':
    plot_class_distribution_sailing_dataset()

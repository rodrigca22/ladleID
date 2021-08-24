from configparser import ConfigParser

### APP CONFIGURATION LOAD ##
config = ConfigParser()


def writeDefaultConfig():
    config['settings'] = {
        'debug': 'false',
        'dataDumpEnabled': 'False',     # Stores images from video feed every 1 min and writes debug data on s CSV file
        'leftScanEnabled': 'False',
        'rightScanEnabled': 'False',
        'colorFilterON': 'True',
        'usePresetHSVFilter': 'True'
    }

    config['HSV_Filter'] = {
        "hueMin": "0",
        "hueMax": "179",
        "satMin": "0",
        "satMax": "72",
        "valMin": "65",
        "valMax": "176"
    }
    config['opc-ua_server'] = {
        'enable_OPC_Server': 'True',
        'endpoint': '127.0.0.1',
        'port': '5000'
    }

    config['video_feed'] = {
        'url': 'rtsp://10.81.98.80/?line=4?inst=2',
        'frame_delay': '500'
    }

    config['neural_network'] = {
        'minCertainty': '0.9',
        'validationSamples': '50'
    }

    config['image_processing'] = {
        'scaleFactor': '10',
        'thresholdladleLeft': '220',
        'thresholdladleRight': '180'
    }

    config['box_coordinates'] = {
        'x1': '100',
        'y1': '310',
        'w1': '100',
        'h1': '100',
        'x2': '530',
        'y2': '330',
        'w2': '100',
        'h2': '100'
    }

    config['single_digit_boxes'] = {
        'minSnglDigitBoxWidth': '100',
        'maxSnglDigitBoxWidth': '250',
        'minSnglDigitBoxHeigth': '200',
        'maxSnglDigitBoxHeigth': '350'
    }

    config['double_digit_boxes'] = {
        'minDblDigitBoxWidth': '320',
        'maxDblDigitBoxWidth': '500',
        'minDblDigitBoxHeigth': '250',
        'maxDblDigitBoxHeigth': '450',
    }

    config['box_filter_params'] = {
        'minFillDregree': '0.2',
        'maxFillDegree': '0.9'
    }



    with open('config.ini','w') as f:
        config.write(f)


if __name__ == __name__: writeDefaultConfig()
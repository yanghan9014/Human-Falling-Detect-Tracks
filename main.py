from fall_detector import Fall_detector
import argparse

if __name__ == "__main__":
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('--index', default=0,  # required=True,
                        help='Street lamp index')
    par.add_argument('--source_url', default=8888,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--target_url', default='http://10ba-125-227-134-216.ngrok.io/api/fall', type=str, required=True,
                        help='output post url')
                        
    par.add_argument('--detection_input_size', type=int, default=192,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default="./tmp.mp4",
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    detector = Fall_detector(args)
    detector.load_models()
    detector.detect()

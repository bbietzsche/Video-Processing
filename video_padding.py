import cv2
import os
import numpy as np

def modify_video_total_frames(input_path, output_path, desired_total_frames, target_width, target_height):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)


    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == desired_total_frames:
        #print("The video already has the desired total frames.")
        # If the video already has the desired total frames, we don't need to modify it.
        # However, if you want to resize it and save it with the given target width and height, you can add the following lines:
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video resized to {target_width}x{target_height} and saved to {output_path}.")
        return

    if total_frames < desired_total_frames:
        # Zero-padding the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        for frame in frames:
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)

        # Repeat the black frame to add zero-padding
        num_padding_frames = desired_total_frames - total_frames
        zero_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)  # Create black frame with the desired shape
        for _ in range(num_padding_frames):
            out.write(zero_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video resized to {target_width}x{target_height} and padding added to {desired_total_frames} frames.")

    else:
        # Trimming and resizing the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        num_trim_frames = total_frames - desired_total_frames
        for _ in range(desired_total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (target_width, target_height))
            out.write(resized_frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video trimmed to {desired_total_frames} frames and resized to {target_width}x{target_height}.")
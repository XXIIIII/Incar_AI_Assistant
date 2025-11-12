import torch
from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array
import torch.nn.functional as F
import numpy as np

class MomentDETRPredictor:
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda"):
        self.clip_len = 2  # seconds
        self.device = device
        # Loading feature extractors...
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        # Loading trained Moment-DETR model...
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding of this pretrained MomentDETR only support video up " \
                               "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )

        # decode outputs
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in MomentDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames, scores for each frame
            )
            predictions.append(cur_query_pred)

        return predictions


def catch_highlight(query_text, input_video_path, save_video_path,query_path=None, num_hd=1):
    # load example data
    from utils.basic_utils import load_jsonl
    if query_path is None:
        query_path = "run_on_video/example/queries.jsonl"
    queries = load_jsonl(query_path)
    # query_text_list = [e["query"] for e in queries]
    query_text_list = [query_text]
    ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"

    # run predictions
    clip_model_name_or_path = "ViT-B/32"
    # clip_model_name_or_path = "tmp/ViT-B-32.pt"
    moment_detr_predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device="cuda"
    )
    predictions = moment_detr_predictor.localize_moment(
        video_path=input_video_path, query_list=query_text_list)

    # save_frame_from_video(video_path=video_path, time_in_seconds=predictions[0]['pred_relevant_windows'])
    for i in range(num_hd):
        start_time = int(predictions[0]['pred_relevant_windows'][i][0])
        end_time = int(predictions[0]['pred_relevant_windows'][i][1])
        #print(f'start saving video:{save_video_path}')
        save_photo_time = (start_time + end_time) // 2
        #print(save_photo_time )
        #print("\nHELLO WORLD\n")
        save_picture_from_video(input_video_path, save_photo_time , output_image_path= f'./{save_video_path}.png')
        # save_video_clip(input_video_path, start_time, end_time, output_video_path=save_video_path)
    

    # print data
    # for idx, query_data in enumerate(queries):
    #     print("-"*30 + f"idx{idx}")
    #     # print(f">> query: {query_data['query']}")
    #     # print(f">> video_path: {video_path}")
    #     # print(f">> GT moments: {query_data['relevant_windows']}")
    #     print(f">> Predicted moments ([start_in_seconds, end_in_seconds, score]): "
    #           f"{predictions[idx]['pred_relevant_windows']}")
    #     # print(f">> GT saliency scores (only localized 2-sec clips): {query_data['saliency_scores']}")
    #     print(f">> Predicted saliency scores (for all 2-sec clip): "
    #           f"{predictions[idx]['pred_saliency_scores']}")
        
from moviepy.video.io.VideoFileClip import VideoFileClip
def save_video_clip(source_video_path, start_time, end_time, output_video_path):
    # Load the source video
    video = VideoFileClip(source_video_path)
    
    # Cut out the subclip from the source video
    clip = video.subclip(start_time, end_time)
    
    # Write the resulting clip to a file
    clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    
    # Close the clip to release resources
    clip.close()
    video.close()
    
    #print(f"Clip from {start_time}s to {end_time}s saved as {output_video_path}")


def slide_video_and_catch_hd(source_video_path, slide_video_path='./hd.mp4'):
    # Load the source video
    video = VideoFileClip(source_video_path)
    max_time = video.duration
    #print(max_time)
    
    # Cut out the subclip from the source video
    time, count = 0, 0
    while(True):
        end_time = time+150 if time+150 < max_time else max_time
        clip = video.subclip(time, end_time)
        # Write the resulting clip to a file
        clip.write_videofile(slide_video_path, codec='libx264', audio_codec='aac')
        #print(f'success to save slide video{slide_video_path}')
        query_text = 'a different view with special buildings'
        #promt try : "a different view with special buildings"
        #          : "a different view with tower : not good"
        
        #print(time)
        catch_highlight(query_text, slide_video_path, save_video_path=f'./{count}')
        time+=150
        count+=1

        if end_time >= max_time:
            break
    # Close the clip to release resources
    clip.close()
    video.close()


import cv2

# def save_frame_from_video(video_path, time_in_seconds, output_image_path='./hd.jpg'):
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print("Error: Could not open video.")

#     # Get the frames per second (fps) of the video
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Calculate the frame number to capture based on the time and fps
#     frame_number = int(time_in_seconds * fps)
    
#     # Set the video position to the frame number
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
#     # Read the frame
#     success, frame = cap.read()
    
#     if not success:
#         print("Error: Could not read frame from video.")
#         cap.release()
    
#     # Save the frame as an image file
#     cv2.imwrite(output_image_path, frame)
    
#     # Release the video capture object
#     cap.release()
#     print(f"Frame at {time_in_seconds}s saved as {output_image_path}")

#try
def save_picture_from_video(video_path, save_time , output_image_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the frame number to capture based on the time and fps
    frame_number = int(save_time * fps)
    
    # Set the video position to the frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    success, frame = cap.read()
    
    if not success:
        print("Error: Could not read frame from video.")
        cap.release()
    
    # Save the frame as an image file
    cv2.imwrite(output_image_path, frame)
    
    # Release the video capture object
    cap.release()
    #print(f"Frame at {save_time}s saved as {output_image_path}")

if __name__ == "__main__":
    video_path = "./test.mp4"
    slide_video_and_catch_hd(video_path)

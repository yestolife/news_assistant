import gradio as gr
from modelscope import AutoModelForCausalLM, AutoTokenizer


# def load_model():
#     model = AutoModelForCausalLM.from_pretrained(
#         "qwen/qwen-1_8b-chat",
#         trust_remote_code=True,
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-1_8b-chat",trust_remote_code=True)
#     return model, tokenizer


def news_writer(prompt): 
    # messages = [
    #     {"role": "system", "content": "你是一个新闻稿编制助手"},
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # generated_ids = model.generate(
    #     model_inputs.input_ids,
    #     max_new_tokens=512
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = 'news'
    return response

    
# model, tokenizer = load_model()
# device = "cuda"

webui = gr.Interface(
    news_writer, 
    inputs=[gr.Textbox(label="输入主要内容", lines=5)],
    outputs=[gr.Textbox(label="生成新闻", lines=5)],
    title="新闻写作助手",
    allow_flagging='never') 

webui.launch(share=True)

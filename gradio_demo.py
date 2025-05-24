# import gradio as gr

# def greet(name):
#     return '你好' + name
#
# demo = gr.Interface(
#     fn=greet,
#     inputs=gr.Text(label="姓名",value="吴小辉",lines=5),
#     outputs=gr.Text(label="输出",lines=5),
# )
#
# if __name__ == "__main__":
#     demo.launch()  # 正确的拼写是 launch()

import gradio as gr
import time

def nvshen_bot(message, history):
    pos = message.find('吗')
    time.sleep(1)
    if pos != -1:
        return message[:pos]
    else:
        return '嗯'

css = '''
.gradio-container { max-width:850px !important; margin:20px auto !important;}
.message { padding: 10px !important; font-size: 14px !important;}
'''

demo = gr.ChatInterface(
    css=css,
    fn=nvshen_bot,
    title='女神聊天机器人',
    chatbot=gr.Chatbot(height=400, bubble_full_width=False),
    theme=gr.themes.Default(spacing_size='sm', radius_size='sm'),
    textbox=gr.Textbox(placeholder="开始跟女神聊天吧~", container=False, scale=7),
    examples=['在吗', '饿了吗？', '一起去吃点？', '等我去接你'],
    submit_btn=gr.Button('提交', variant='primary'),
)

if __name__ == "__main__":
    demo.launch(debug=True)


























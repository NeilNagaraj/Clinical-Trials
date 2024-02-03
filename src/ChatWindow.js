import React, { Component } from 'react'
import { Launcher } from 'react-chat-window'


class Demo extends Component {

    constructor() {
        super();
        this.state = {
            messageList: []
        };
    }
    
    _onMessageWasSent(message) {
        console.log(message);
        var messageText = { "text": message["data"]["text"] };
        var typing = 'div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>'
        console.log(messageText);
        this.setState({
            messageList: [...this.state.messageList,message,{
                author: "them",
                type: "text",
                data: {
                    "text": "typing...",
                }
            }]
        })
        fetch('http://localhost:5003/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(messageText)
        })
            .then(response => response.json())
            .then(data => {
                console.log("Returned data", data);
                var messagesList = []
                data["messages"].reverse();
                for (var i = 0; i < data["messages"].length; i++) {
                    if (i % 2 == 0) {
                        messagesList.push({
                            author: "me",
                            type: "text",
                            data: {
                                "text": data["messages"][i],
                            }
                        });
                    }
                    else { 

                        messagesList.push({
                            author: "them",
                            type: "text",
                            data: {
                                "text": data["messages"][i],
                            }
                        });

                    }
                }
                console.log(messagesList)
                this.setState({
                    messageList: messagesList
                })
            }
                )
            .catch(error => console.error(error));
      
    }

    _sendMessage(text) {
       
        if (text.length > 0) {
            this.setState({
                messageList: [...this.state.messageList, {
                    author: 'them',
                    type: 'text',
                    data: { text }
                }]
            })
        }
    }

    render() {
        return (<div>
            <Launcher
                agentProfile={{
                    teamName: 'Clinical Trial Assistant',
                    // imageUrl: 'https://a.slack-edge.com/66f9/img/avatars-teams/ava_0001-34.png'
                }}
                onMessageWasSent={this._onMessageWasSent.bind(this)}
                messageList={this.state.messageList}
                showEmoji
            />
        </div>)
    }
}

export default Demo;
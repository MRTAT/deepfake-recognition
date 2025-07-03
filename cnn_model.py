import torch.nn as nn
import torch

class Cnn(nn.Module):

    # Input size (3, 224, 224)
    def __init__(self, num_classes = 2):
        super().__init__()

        # DEFINE
        self.conv1 = self.make_block(in_channels=3, out_channels=8)
        self.conv2 = self.make_block(in_channels=8, out_channels=16)
        self.conv3 = self.make_block(in_channels=16, out_channels=32)
        self.conv4 = self.make_block(in_channels=32, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=128)  # images size : 224 / 32 = 7  -> 128 * 7 * 7
        # END DEFINE

        # GRU
        self.gru = nn.GRU(
            input_size=128, # Output channel of CNN
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        # END GRU

        # ATTENTION
        # Calculate the important score for each time step
        # Attention mechanism
        gru_output_size = 256 * 2  # * 2 if bidirectional
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.Tanh(),
            nn.Linear(gru_output_size // 2, 1)
        )


        # FCN
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),

            # in_features = hidden_size of GRU  * 2 , if use bidirectional
            nn.Linear(in_features=256 * 2, out_features=512),  # FC1 128*7*7
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=512, out_features=1024),   # FC4
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(in_features=1024, out_features=num_classes)  # FC7
        )
        # END FCN

    # CNN
    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same"),  # conv 1.0
            nn.BatchNorm2d(num_features=out_channels),  # conv 1.1
            nn.LeakyReLU(), # no change

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same"),  # conv 1.3
            nn.BatchNorm2d(num_features=out_channels),   # conv 1.4
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2)
        )
    # END CNN

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Reshape for GRU
        x = x.permute(0, 2, 3, 1)  # B x 128 x 7 x 7  -> B x 7 x 7 x 128
        x = x.reshape(x.size(0), -1, 128)  # B * 49 * 128

        gru_out, _ = self.gru(x) # [B, 49, 256]  | out, _ : _ using for skip h_n (hidden state)  just take
        # out = out[:, -1, :]  # (last time step) → [B, 256]

        # Attention mechanism
        attention_scores = self.attention(gru_out)  # [B, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum
        context_vector = torch.sum(attention_weights * gru_out, dim=1)  # [B, hidden*2]

        # Classification
        output = self.fc(context_vector)

        return output


if __name__ == '__main__':
    model = Cnn()
    input_data = torch.rand(16, 3, 224, 224)
    if torch.cuda.is_available():
        model.cuda()  # in_place function
        input_data = input_data.cuda()
    result = model(input_data)  # <=> call forward
    pred = torch.argmax(result, dim=1)  # lấy class có giá trị lớn nhất
    print(pred)  # tensor([1, 0, 1, ..., 0])

    print(result.shape)

    # for name, param in model.named_parameters():
    #     print(name, param)
    # print(result.shape)  # (16, 8, 222, 222) (B x C x H x W) : 8 -- out_channel : 222 -- kernel = 3 then decreasing 2 unit
    # if want output same size input then have 2 ways -> increase: padding = "same" |

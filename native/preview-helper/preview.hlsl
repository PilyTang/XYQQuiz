Texture2D<float4> Source : register(t0);
SamplerState LinearSampler : register(s0);

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VertexOutput VSMain(uint id : SV_VertexID) {
    VertexOutput output;
    output.uv = float2((id << 1) & 2, id & 2);
    output.position = float4(
        output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0),
        0.0,
        1.0);
    return output;
}

float4 PSMain(VertexOutput input) : SV_Target {
    return Source.Sample(LinearSampler, input.uv);
}

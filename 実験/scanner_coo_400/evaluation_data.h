#pragma once

// dimention_40/40_dimention_before.ppm
// 左上から行順、RGB順。各成分を255で正規化。
constexpr size_t EVALUATION_WIDTH = 3;
constexpr size_t EVALUATION_HEIGHT = 3;
constexpr size_t EVALUATION_PIXEL_COUNT = 9;

const char EVALUATION_SOURCE_SHA256[] =
    "aef4bbf02c893cd55ca2faee519486b27311c804276259ba551d7c5a0714b562";

const float evaluationInputs[EVALUATION_PIXEL_COUNT][3] = {
    {195.0f/255.0f,  52.0f/255.0f,  24.0f/255.0f},
    { 61.0f/255.0f, 248.0f/255.0f,  48.0f/255.0f},
    { 21.0f/255.0f,  73.0f/255.0f, 104.0f/255.0f},
    {245.0f/255.0f, 253.0f/255.0f,  72.0f/255.0f},
    { 72.0f/255.0f, 252.0f/255.0f, 217.0f/255.0f},
    { 96.0f/255.0f,  61.0f/255.0f,  81.0f/255.0f},
    {  2.0f/255.0f,   0.0f/255.0f,   1.0f/255.0f},
    { 56.0f/255.0f, 126.0f/255.0f,  69.0f/255.0f},
    {255.0f/255.0f, 255.0f/255.0f, 253.0f/255.0f},
};

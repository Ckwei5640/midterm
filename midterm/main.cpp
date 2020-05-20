#include "DA7212.h"
#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "mbed.h"
#include "uLCD_4DGL.h"
#include <string>
#include <cmath>

DA7212 audio;
uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;
int16_t waveform[kAudioTxBufferSize];
InterruptIn sw3(SW3);
InterruptIn sw2(SW2);
DigitalOut gled(LED2);
Serial pc(USBTX, USBRX);
EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue gqueue(32 * EVENTS_EVENT_SIZE);
EventQueue mqueue(32 * EVENTS_EVENT_SIZE);
Thread t_play;
Thread t_game;
Thread t_mode(osPriorityHigh);
Thread t_DNN(osPriorityNormal, 100*1024/*120K stack size*/);
int current_gesture = -1;
int song_select = 1;
int current_song = 0;
int mode_select = 1;
int current_mode = 0;
char buff;
int j;
int song = 2;

std::string name[3] = {"LittleStar", "Perfect", ""};
std::string mode[5] = {"Forward", "Backward","Song Selection","Add Song","Taiko"};
int song1[43] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261, 0};
int noteLength1[43] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2, 1};
int song2[48] = {
  294, 330, 294, 261, 523, 494, 440,
  494, 330, 392, 349, 330, 349, 330,
  523, 494, 440, 494, 330, 392, 392,
  392, 392, 440, 330, 294, 261, 261,
  523, 494, 440, 494, 330, 392, 349,
  330, 349, 330, 294, 349, 330, 330, 294, 294, 261, 247, 261, 0};
int noteLength2[48] = {
  2, 1, 1, 2, 1, 1, 1,
  1, 2, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 2, 1, 1,
  1, 1, 1, 1, 1, 1, 1,
  2, 1, 1, 1, 2, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1};
int song3[62];
int noteLength3[62];

void playNote(int freq){
  for (int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  // the loop below will play the note for the duration of 1s
  for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
  {
    audio.spk.play(waveform, kAudioTxBufferSize);
  }
}

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }
  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;
  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void DNN(){

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;
  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  /*if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }*/
  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE());

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;
  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  /*if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }*/
  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  /*if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }*/
  error_reporter->Report("Set up successful...\n");

  while (100) {
    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }
    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);
    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      //error_reporter->Report(config.output_message[gesture_index]);
      current_gesture = gesture_index;
    }
  }
}

void CanSwitch(){ //switch mode
  mode_select = !mode_select;
}

void ModeSwitch(){
  uLCD.cls();
  uLCD.printf("\nMode Selecting...\n");
  while(1){
    wait(0.01);
    if(mode_select == 0) break;
    if(current_gesture == 2){//waveleft
      if(current_mode == 0) current_mode = 4;
      else current_mode--;
      current_gesture = -1;
      uLCD.printf("  %s\n",mode[current_mode].c_str());
    }
    if(current_gesture == 3){//waveright
      if(current_mode == 4) current_mode = 0;
      else current_mode++;
      current_gesture = -1;
      uLCD.printf("  %s\n",mode[current_mode].c_str());
    }
    //sw2.fall(CanSwitch);
  }

}

void SelectMode(){ //mode song selection
  song_select = !song_select;
}

void SongSwitch(){
  uLCD.cls();
  uLCD.printf("\nSong Selecting...\n");
  while(1){
    wait(0.01);
    if(current_gesture == 2){//waveleft
      if(current_song == 0) current_song = 0;
      else current_song--;
      current_gesture = -1;
      uLCD.printf("  %s\n",name[current_song].c_str());
    }
    if(current_gesture == 3){//waveright
      if(current_song == song) current_song = song;
      else current_song++;
      current_gesture = -1;
      uLCD.printf("  %s\n",name[current_song].c_str());
    }
    if(song_select == 0) break;
    //sw3.fall(SelectMode);
  }
}

void StopMusic(){
  audio.spk.pause();
  //waveform = (int16_t)0;
  uLCD.cls();
  uLCD.filled_rectangle(32,32,96,96,WHITE);
  wait(1);
}

void PlaySong(){
    if(current_song == 0){
      //uLCD.printf("\nplaying123\n");
      //uLCD.printf("\n    %s \n",name[current_song].c_str());
      for(int i = 0; i < 43; i++){
          int length = noteLength1[i];
          while(length--)
          {
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
            {
              if(current_gesture == 0) {
                StopMusic();
              }
              queue.call(playNote, song1[i]);
            }
            if(length < 1) wait(0.9);
          }
          audio.spk.pause();
        }
      audio.spk.pause();
    }
    
    if(current_song == 1){
      //uLCD.printf("\nplaying...\n");
      //uLCD.printf("\n    %s \n",name[current_song].c_str());
      for(int i = 0; i < 48; i++){
          int length = noteLength2[i];
          while(length--)
          {
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
            {
              if(current_gesture == 0) {
                StopMusic();
              }
              queue.call(playNote, song2[i]);
            }
            if(length < 1) wait(0.7);
          }
          audio.spk.pause();
        }
      audio.spk.pause();
    }
    if(current_song == 2){
      //uLCD.printf("\nplaying...\n");
      //uLCD.printf("\n    %s \n",name[current_song].c_str());
      for(int i = 0; i < 62; i++){
          int length = noteLength3[i];
          while(length--)
          {
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
            {
              if(current_gesture == 0) {
                StopMusic();
              }
              queue.call(playNote, song3[i]);
            }
            if(length < 1) wait(0.5);
          }
          audio.spk.pause();
        }
      audio.spk.pause();
    }
}

void AddSong(){
  uLCD.cls();
  uLCD.printf("\nAdding Song...\n");
  gled = 0;
  while(1){
    if(pc.readable()){
      buff = pc.getc();
      if(buff == '\n') break;
      else name[2] = name[2] + buff;
    }
  }

  for(int i = 0; i < 62; i++){
    j = 2;
    while(1){
      if(pc.readable()){
        buff = pc.getc();
        if(buff == '\n') break;
        else{
          song3[i] = song3[i] + pow(10,j)*(buff - 48);
          j--;
        }
      }
    }
  }

  for(int i = 0; i < 62; i++){
    while(1){
      if(pc.readable()){
        buff = pc.getc();
        if(buff == '\n') break;
        else noteLength3[i] = buff - 48;
      }
    }
  }

  gled = 1;
  uLCD.printf("  %s\n",name[2].c_str());
  song++;
  wait(1);
}

void Taiko(){
  int score = 0;
  //current_song = 0;
  for(int i = 0;i < 42;i++){
    uLCD.cls();
    uLCD.printf("Score:%d",score);
    if(noteLength1[i] != 1) uLCD.filled_circle(64, 64, 20, BLUE);
    else uLCD.filled_circle(64, 64, 15, RED);
    if((noteLength1[i] == 1 && current_gesture == 2) || (noteLength1[i] != 1 && current_gesture == 3)){
      score++;
      current_gesture = -1;
    }
    wait(0.7);
  }
  uLCD.cls();
  uLCD.text_width(2);
  uLCD.text_height(2);
  uLCD.printf("\n\n\nScore:%d",score);
}

int main() {
    uLCD.cls();
    uLCD.color(GREEN);
    uLCD.text_width(2);
    uLCD.text_height(2);
    uLCD.printf("\n\n\nMusic\n  Player\n");
    uLCD.text_width(1);
    uLCD.text_height(1);
    uLCD.color(RED);
    uLCD.printf("&TAIKO");
    wait(4);
    uLCD.color(GREEN);
    gled = 1;

    t_DNN.start(DNN);
    while(1){
      t_mode.start(callback(&mqueue, &EventQueue::dispatch_forever));
      t_play.start(callback(&queue, &EventQueue::dispatch_forever));
      t_game.start(callback(&gqueue, &EventQueue::dispatch_forever));
    
      sw2.fall(CanSwitch);
      sw3.fall(SelectMode);
      mqueue.call(ModeSwitch);
      while(mode_select){
        wait(0.01);
      }
      //uLCD.cls();
      if(current_mode == 3 && mode_select == 0) AddSong();
      if(current_mode == 0 && mode_select == 0){
        current_song = 0;
        for(int i = current_song;i < song;i++){
          uLCD.cls();
          uLCD.printf("\nplaying...\n");
          uLCD.printf("\n    %s \n",name[current_song].c_str());
          PlaySong();
          current_song++;
        }
      }
      if(current_mode == 1 && mode_select == 0){
        current_song = song-1;
        for(int i = current_song;i >= 0;i--){
          uLCD.cls();
          uLCD.printf("\nplaying...\n");
          uLCD.printf("\n    %s \n",name[current_song].c_str());
          PlaySong();
          current_song--;
        }
      }
      if(current_mode == 4 && mode_select == 0){
        current_song = 0;
        gqueue.call(Taiko);
        PlaySong();
      }
      if(current_mode == 2 && mode_select == 0) queue.call(SongSwitch);
      while(song_select){
        wait(0.01);
      }
      if(current_mode == 2 && mode_select == 0){
          uLCD.cls();
          uLCD.printf("\nplaying...\n");
          uLCD.printf("\n    %s \n",name[current_song].c_str());
          PlaySong();
      }
      mode_select = 1;
      song_select = 1;
    }
    

}
# 42CPNN



## Overview

I have always been fascinated by predictions.  
Thank God for the modern era, where I can Nostradamus anything whil being backed by technology.  
This repository contains a very basic Convolutional Neural Network (badly) written in Python, able to make (somewhat) reliable predictions on your acceptation at 42 Paris following your piscine.

## Foreword

This is no way, shape, or form, a reliable tool, nor should it be used to correctly assess or predict for real your entry at the school.
This project is solely an education one, and by no mean a way to bypass or influence the current system in place for the admissions at 42 Paris.
The results from the model should not be trusted. I cannot be liable for any wrong prediction from the model, or any usage of it by anyone. 

My goal by providing this model is to share my love for data analysis and machine learning.  
It was trained on different profiles from people who attended the C Piscine at 42 Paris.  
During this training, I realised a lot of data cleanup was necessary, due to some values being weirdly out of place.  
Those values are mostly the result of the policy of 42 Paris for inclusion and egality.  
I will say it now:  
**Those policies are good for the software engineering environment**
I absolutely support the inclusion of historic minorities in our beautiful community of software engineers, and, although at times they might be a bit too blunt, those policies are the way to go.
If you think those policies are not a good thing, let's agree to disagree, but you must NEVER use this model to discriminate, in any way, against anyone or any group of people. DO NOT use this model to make some weid sexist or racist statistics about "who should have been rejected but was accepted". It does not make you cool to do that, it just makes you a weird loser.


## Methodology

Coming soon

## Usage

To run this project, you will need Python 3 (>= 3.10) and the numpy package (>= 2.2.3)
First, clone the repository
```
git clone <repo_url>
cd 42CPNN
```

If Python and numpy aren't already installed, go ahead and install them 
 - See https://www.python.org/downloads/ for Python
 - See https://numpy.org/ for Numpy

Make sure the file `params.bin` is in the same directory as the `test.py` file.  
You can now execute the folowing command to check for the different options of the project
```
python3 test.py --help
```

Enjoy.

## Licensing

```
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
```

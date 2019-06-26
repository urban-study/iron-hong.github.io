---
title:  "C++ practice"
categories: post
mathjax: true
---
I try to collect C++ references which I keep forgetting. 


- Class: class is a [custom datatype](https://youtu.be/-EwsSCObiRw). It could be a data or a function. 
- Pointer: pointer is an address of a variable, but why [do you need pointer?](https://youtu.be/egXLylrJeic) and [how do you use it](https://youtu.be/UCWWObpNUZw)

```
#include <iostream>

int main()
{
    int num = 342;
    std::cout<<"num = "<<num<<"\n";
    std::cout<<"address of num is at &num = "<< &num<<"\n";
    return 0;
}
```
```
num = 342
address of num is at &num = 0x7ffd2a738f6c
```

- [Reference](https://classroom.udacity.com/courses/ud999)

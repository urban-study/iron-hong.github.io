---
title:  "C++ practice"
categories: post
mathjax: true
---
I try to collect C++ syntax while I am practicing [Unreal C++ implementation](https://www.udemy.com/unrealengine-cpp/).  

- Class: class is a [custom datatype](https://youtu.be/-EwsSCObiRw). It could be a data or a function. 
- Pointer: pointer is an address of a variable, but why [do you need pointer?](https://youtu.be/egXLylrJeic) and [how do you use it?](https://youtu.be/UCWWObpNUZw)

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
- Dereferencing: if you want to refer to the value based on the address, it's called dereferencing. 
```
#include <iostream>
int main()
{
    int num = 32;
    int * pointerTonum = &num;
    std::cout<<"pointerTonum stores the address of num as"<< pointerTonum<<endl;
    std::cout<<"the address of PointerTonum stores the value of num as<< *pointerTonum<<endl;
}
```
- To access a member in a class, use -> . [For instance](http://www.cplusplus.com/forum/beginner/53293/), foo->bar means *foo.bar.
- To access a function in a class, use :: .
- Constructor is used to set the intial value of the class. i.e. ClassName::ClassName()
- #pragma once is to cause the current source file to be included only once in a single compilation.
- a class (A) after the [single colon](http://www.cplusplus.com/forum/beginner/235722/) from another class (B) means, 
B is the inherited class from A. Here is the example from Unreal source code:
```
class ENGINE_API USphereComponent : public UShapeComponent
{
    // code
}
```
- [Overloading](https://www.tutorialspoint.com/cplusplus/cpp_overloading.htm): C++ allows the same function name could be used for multiple implementation. 

```
#include <iostream>
using namespace std;
class printNum{
    public:
        void print(int i){
            cout<<"print integer: "<< i <<endl;
        }
        void print(double f){
            cout<<"print float: "<< f <<endl;
        }
        void print(char*c){
            cout<<"print character: "<< c <<endl;
        }

}

int main()
{
    printNum pn;

    // print integer
    pn.print(23);
    // print float
    pn.print(234.23);
    // print character
    pn.print("hello C++");

    return 0;
}
```

- [Template](https://en.m.wikipedia.org/wiki/Template_(C%2B%2B)): it allows a function or class to work on many different data types without being rewritten for each one. For instance, ```foo<T>```is a template and T is the template parameter. 

```
void AFPSObjectiveActor::NotifyActorBeginOverlap(AActor* OtherActor)
{
	Super::NotifyActorBeginOverlap(OtherActor);
	PlayEffects();

	AFPSCharacter* MyCharacter = Cast<AFPSCharacter>(OtherActor);
	if (MyCharacter)
	{
		MyCharacter->bIsCarryingObjective = true;
	}

}
```


- Reference 
    
    - [Udacity C++ free course](https://classroom.udacity.com/courses/ud999)
    - [Unreal C++ implementation](https://www.udemy.com/unrealengine-cpp/)
    

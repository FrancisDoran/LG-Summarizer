### Link Grammar API Documentation 
*\(Relevant to this project only\)*

<hr style="height:4px; solid gray">  

#### Structure
* Class  
    |_ Attributes  
    |_ Methods

**Only attributes and methods I think might be necessary are noted.**  
For more details or to look at other methods look [here.](https://github.com/opencog/link-grammar/blob/master/bindings/python/linkgrammar.py)

#### TOC
* [Docs layout](####structure)
* [Class: ParseOptions](#####class-parseoptions)
* [Class: Link](#####class-link)
* [Class: Linkage](#class-linkage)
* [Class: Sentence](#class-sentence)

<hr style="height:4px; solid black">

##### Class: ParseOptions
  
Configuration class for the parser itself.

##### Attributes
* linkage\_limit  
default = 1000
  
* min\_null\_count
default = 0 (no minimum)
  
* max\_null\_count
default = 0 (no maximum)
  
* display\_morphology
default = True
  
* islands\_ok  
default = False  
Islands are disconnected link graphs  
in the same sentence.

<hr style="height:4px; solid black">

##### Class: Link

Defines a single word's link.

##### Attributes

* index  
Position of the link.

* left\_word
* left\_label

* right\_word
* right\_label

##### Methods

* num\_domains\(\)  
The "area" of a sentence a link "belongs" to.

<hr style="height:4px; solid black">

##### Class: Linkage

A set of [Link](#####class-link)\(s\). Usually contains a single parse of  
a sentence where each sentence can have multiple parses.

##### Attributes

* sentence  
sentence the link is a part of.

* parse\_options  
The parse options used for the link parse  
of a sentence. 

##### Methods

* link\(\)  
Links a given word in a sentence.

<hr style="height:4px; solid black">

##### Class: Sentence

The sentence object is used as input to the parser.  
Given a Sentence object, the parser will produce  
the set of possible linkages. And each linkage contains  
the links between the words in the sentence.

##### Methods
* parse\(\)
Returns the linkages found from the sentence object  
it was called on.

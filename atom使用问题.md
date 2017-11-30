[toc]
# atom使用
## 墙内配置插件安装镜像
更换镜像方法：
1. 在路径 C:\Users\hn\.atom下新建文本文档，命名 .atomrc
  内容：(注意：使用notepad++ 更改文件名，使用记事本出错)
  registry=https://registry.npm.taobao.org
  strict-ssl = false
2. 命令行，检查是否能够正常安装。
  apm install --check 结果为 done。成功。

## snippets 代码段自定义
  snippets[^1]
1. 路径 C:\Users\hn\.atom\snippets.cson
2. 自定义代码段：
  atom使用cson来定义snippet，scope可以参见snipppet,多行代码使用'''code line here'''

  ```
  '.source.coffee':           填写scope的地方
  'Console log':              要新建的 snippet 的名称
    'prefix': 'log'           触发当前 snippet 的代码
    'body': 'console.log $1'  要填充的代码
  ```

3. 实例
    ```
    '.source.c':
    'if':
      'prefix': 'if'
      'body': 'if(\$1){\$2}'
    ```
    \$1 表示光标的默认位置
    \$2 按下 tab 键，光标跳到的第二个位置，以此类推
    另外，使用\${1:'replaced'}替换\$1即可转换光标到选定状态，选定内容为replaced
4. 各种编程语言的scope：
  ```
  ActionScript: source.actionscript.2
  AppleScript: source.applescript
  ASP: source.asp
  Batch FIle: source.dosbatch
  C#: source.cs
  C++: source.c++
  Clojure: source.clojure
  CoffeeScript: source.coffee
  CSS: source.css
  D: source.d
  Diff: source.diff
  Erlang: source.erlang
  Go: source.go
  GraphViz: source.dot
  Groovy: source.groovy
  Haskell: source.haskell
  HTML: text.html(.basic)
  JSP: text.html.jsp
  Java: source.java
  Java Properties: source.java-props
  Java Doc: text.html.javadoc
  JSON: source.json
  Javascript: source.js
  BibTex: source.bibtex
  Latex Log: text.log.latex
  Latex Memoir: text.tex.latex.memoir
  Latex: text.tex.latex
  LESS: source.css.less
  TeX: text.tex
  Lisp: source.lisp
  Lua: source.lua
  MakeFile: source.makefile
  Markdown: text.html.markdown
  Multi Markdown: text.html.markdown.multimarkdown
  Matlab: source.matlab
  Objective-C: source.objc
  Objective-C++: source.objc++
  OCaml campl4: source.camlp4.ocaml
  OCaml: source.ocaml
  OCamllex: source.ocamllex
  Perl: source.perl
  PHP: source.php
  Regular Expression(python): source.regexp.python
  Python: source.python
  R Console: source.r-console
  R: source.r
  Ruby on Rails: source.ruby.rails
  Ruby HAML: text.haml
  SQL(Ruby): source.sql.ruby
  Regular Expression: source.regexp
  RestructuredText: text.restructuredtext
  Ruby: source.ruby
  SASS: source.sass
  Scala: source.scala
  Shell Script: source.shell
  SQL: source.sql
  Stylus: source.stylus
  TCL: source.tcl
  HTML(TCL): text.html.tcl
  Plain text: text.plain
  Textile: text.html.textile
  XML: text.xml
  XSL: text.xml.xsl
  YAML: source.yaml
  ```

## latex 环境配置
插件列表[^2]：
* language-latex  LaTeX 命令的自动补全
* atom-latex      在 Atom 中实现 LaTeX 的编译
* pdf-view

## markdown 环境配置
插件列表[^3]:
* Markdown Preview Enhanced

[^1]: http://www.jianshu.com/p/5619189a4b79
[^2]: http://www.latexstudio.net/archives/7017
[^3]: https://shd101wyy.github.io/markdown-preview-enhanced/#/zh-cn/

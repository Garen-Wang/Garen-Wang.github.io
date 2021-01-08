---
title: My Hexo Blog Configuration
tags: blog
mathjax: true
date: 2021-01-08 22:04:28
---

芜湖！起飞！今天备案终于审核通过了！捣鼓了一下，终于把博客弄得像模像样的，就顺带记录一下！

## 序幕

军训汇操在早上结束了，回到宿舍一打开手机就连收到三条信息，终于给爷备案好了！

{% qnimg My-Hexo-Blog-Configuration/WeChat_Image_20210108221637.jpg %}

{% qnimg My-Hexo-Blog-Configuration/WeChat_Image_20210108221657.jpg %}

我啪的一下就开始准备我的博客了，很快啊！

## 博客框架

博客使用Hexo搭建，用了NexT主题(NexT.Gemini)，在GitHub上就能找到这个主题。

Hexo只要有npm就可以安装了，跑条命令安装一下就行。

Hexo的操作可以直接看官方文档，也很容易懂，这里不赘述。

GitHub Pages之前就配置好了，现在主要是需要配置到我的服务器上面去。

## Hexo同步至服务器

首先，在服务器上面安装一下git和nginx。在备案没有成功的时候，可以直接用买服务器时给的弹性公网IP直接去上，效果是一样的。

按照我的印象，当没有安装nginx时，在浏览器中输入ip打开，是会出现小恐龙的，而安装了nginx之后就成了404。这说明nginx确实已经开始起作用了，安装正常。

然后可以在服务器那端用ssh免密登录，粗略流程是这样的：

1. 在本机用`ssh-keygen`创建一个ssh公钥和私钥。
2. 在服务器的`.ssh`目录创建一个`authorized_keys`，再`chmod`一下。
3. 把ssh公钥写到`authorized_keys`上面去。

这个时候，只要本机有私钥，服务器有公钥，我们就可以通过一个ssh命令免密远程登录：

```
$ ssh root@"your_ip"
```

之后创建`/var/repo`文件夹，在里面新建一个叫blog的git仓库，新建命令如下：

```
$ git init --bare blog.git
```

之后，打开`blog.git/hooks/post-receive`，`chmod`一下，同时添加下列内容：

```
#!/bin/sh
git --work-tree=/var/www/hexo --git-dir=/var/repo/blog.git checkout -f
```

之后，在`var/www/hexo`处创建好文件夹，`chmod`一下，这样之后，服务器端的设置就完成了。

最终我们想要的是：在本机输入`hexo d`时，能部署到服务器上，这时需要在根目录下的`_config.yml`下修改：

### 第一处
```
# URL
## If your site is put in a subdirectory, set url as 'http://example.com/child' and root as '/child/'
url: https://your_ip
```

### 第二处
```
deploy:
  - type: git
    repo: git@github.com:Garen-Wang/garen-wang.github.io.git
    branch: master
  - type: git
    repo: root@your_ip:/var/repo/blog.git
    branch: master
```

这样就应该能把hexo部署到你的服务器上面去了。

## 添加备案号

网站还是得加备案号的，不过这里不用改模板，直接在NexT主题的`_config.yml`中修改：

```
footer:
  ...
  
  # Beian ICP and gongan information for Chinese users. See: http://www.beian.miit.gov.cn, http://www.beian.gov.cn
  beian:
    enable: true
    icp: 粤ICP备2021003110号
    # The digit in the num of gongan beian.
    gongan_id:
    # The full num of gongan beian.
    gongan_num: 2021003110
    # The icon for gongan beian. See: http://www.beian.gov.cn/portal/download
    gongan_icon_url: images/beian.png
```

在主题文件夹中的`source`中新建个`images`文件夹，可以把[这张图片](http://www.beian.gov.cn/portal/download)下载到里面去，就可以用相对路径引用了。btw，对头像的设置也是同理。

## mathjax支持

这个东西曾经困扰了我很久，其实只要按下面的顺序，NexT主题也能用上mathjax。

先更换Hexo的Markdown渲染引擎：

```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```

需要在`node_modules/kramed/lib/rules/inline.js`中修改两处（分别是原第11行和第20行）：

```
  //escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
  escape: /^\\([`*\[\]()#$+\-.!_>])/,
```

```
  //em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```

最后在每个需要启用mathjax的博客页面里，在一开始的Front-matter那里加上一句：

```
mathjax: true
```

这样就可以用上LaTeX语法写行内公式和行间公式了。

{% qnimg My-Hexo-Blog-Configuration/2021-01-08_23-29.png %}

## 搜索框

搜索框也很容易实现。

先用npm安装下插件：

```
$ npm install --save hexo-generator-search
$ npm install --save hexo-generator-searchdb
```

在NexT主题文件夹下的`_config.yml`下修改：

```
local_search:
  enable: true
```

重新部署一下之后，就可以看到出现了搜索框。

## 评论系统支持

评论系统中，NexT主题的配置中自带对Valine的支持，我们干脆直接用它咯！

### Valine的使用

1. 在LeanCloud注册
2. 创建应用，名称随意
3. 进入“设置-应用Keys”，获取App ID和AppKey
4. 在主题文件夹中的`_config.yml`修改Valine对应内容为：

```
valine:
  enable: true
  appid: Your leancloud application appid
  appkey: Your leancloud application appkey
  notify: true # Mail notifier
  verify: false # Verification code
  placeholder: Just go go # Comment box placeholder
  avatar: mm # Gravatar style
  guest_info: nick,mail,link # Custom comment header
  pageSize: 10 # Pagination size
  language: zh-cn # Language, available values: en, zh-cn
  visitor: true # Article reading statistic
  comment_count: true # If false, comment count will only be displayed in post page, not in home page
  recordIP: false # Whether to record the commenter IP
  serverURLs: # When the custom domain name is enabled, fill it in here (it will be detected automatically by default, no need to fill in)
  #post_meta_order: 0
```

然后在储存-结构化数据中创建两个新的Class，名称分别为`Comment`和`Counter`，分别可以用来存评论和链接访问数，非常方便。

在LeanCloud后台看到的数据就是这样的：

{% qnimg My-Hexo-Blog-Configuration/2021-01-08_22-44.png %}

之后部署一下就可以看到效果了！

{% qnimg My-Hexo-Blog-Configuration/2021-01-08_22-42.png %}

## 七牛云图床

首先先在博客根目录安装一下需要的Hexo插件：
```
$ npm install --save hexo-qiniu-sync
```

在七牛云右上角的密钥管理就可以找到access key和secret key了，bucket填你自己创建时写的空间名称，在`_config.yml`里面添加这一段配置：

```
qiniu:
  offline: false
  sync: true
  bucket: "your_bucket_name"
  access_key: "your_access_key"
  secret_key: "your_secret_key"
  dirPrefix: static
  urlPrefix: http://"your_qiniu_url"/static
  up_host: http://upload.qiniu.com
  local_dir: static
  update_exist: true
  image: 
    folder: images
    extend: 
  js:
    folder: js
  css:
    folder: css
```

在文档中，就不需要使用Markdown的插入图片格式了，使用下面的格式：

```
{% qnimg test.jpg %}
```

这样的语句会自动读取`static/images/test.jpg`这个路径下的图片。

在更新博客时，可以先跑一下这条命令，将`static/images`下的所有图片都上传到七牛云，这样博客的外链就能访问出图片了。

不过不跑似乎也没关系，在`hexo g`的时候似乎会自动帮你上传，挺贴心的。

## 小彩蛋

### 我大E了啊

在配置的时候有一次跑`hexo g -d`的时候报错了，怎么改都改不好，心态差点崩了，差点要把整个博客重新弄一遍。

这种情况的最好解决方法是一开始就用git维护整个仓库。最后我直接用`git reset`回滚到上次commit的时候，一切就又都回来了。我又继续无止境地配置下去了……

### 什么？DDL？

啊？什么？我今天没赶DDL？

{% qnimg My-Hexo-Blog-Configuration/WeChat_Image_20210108221702.jpg %}

其实明天是数创大作业的deadline。。。

放心，明天弄得完的。deadline是第一生产力。。。

熬夜继续爆肝大作业，还不如早点休息。。。

{% qnimg My-Hexo-Blog-Configuration/84869490_p0.jpg %}

## Reference

https://blog.csdn.net/as480133937/article/details/100138838

https://blog.csdn.net/yexiaohhjk/article/details/82526604

https://zhuanlan.zhihu.com/p/34747279

https://www.jianshu.com/p/70bf58c48010
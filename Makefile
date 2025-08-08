SHELL:=/bin/bash
zs:=$(shell echo 0 {1000..15500..500} 15800 16000 16300 16700 16900 17100 17300 17500 17700)
BASE_URL:=https://heloise.thudep.com:9000/
WGET:=wget --quiet

# 下载数据
.PHONY: datas
datas:=$(zs:%=data/%.h5)
datas: $(datas) concat.h5
data/%.h5:
	@mkdir -p $(@D)
	$(WGET) -P $(@D) $(BASE_URL)$*.h5
# 测试集
concat.h5:
	$(WGET) $(BASE_URL)concat.h5

# 数据预处理
pre/%/s.pq pre/%/t.pq &: geo.h5 $(datas)
	mkdir -p $(@D)
	python3 hist.py -g $< -i $(wordlist 2, $(words $^), $^) -o pre/$*/s.pq pre/$*/t.pq -b $(word 1,$(subst _, ,$*)) -t $(word 2,$(subst _, ,$*))

# 训练
GAM/%/model.rds: pre/%/s.pq pre/%/t.pq
	mkdir -p $(@D)
	./gam.R --ipt $^ --b $(word 1,$(subst _, ,$*)) --t $(word 2,$(subst _, ,$*)) --opt $@ > $@.log

GBM/%/model.pkl: pre/%/s.pq pre/%/t.pq
	mkdir -p $(@D)
	python3 gbm.py -i $^ -o $@ > $@.log

# 默认版本
default_method := GBM
default_bins := 200_300

# 绘图
.PHONY: draw
draw/GAM%.pdf: concat.h5 GAM/%/model.rds
	mkdir -p $(@D)
	python3 draw.py draw --model $(word 2,$^) --concat $< -o $@

draw/GBM%.pdf: concat.h5 GBM/%/model.pkl
	mkdir -p $(@D)
	python3 draw.py draw --type GBM --model $(word 2,$^) --concat $< -o $@

draw: draw/$(default_method)$(default_bins).pdf

# 评分
.PHONY: score
score_GAM%: concat.h5 GAM/%/model.rds
	python3 draw.py validate --model $(word 2,$^) --concat $<

score_GBM%: concat.h5 GBM/%/model.pkl
	python3 draw.py validate --type GBM --model $(word 2,$^) --concat $<

score: score_$(default_method)$(default_bins)

all: draw score
.PHONY: all

clean:
	rm -rf pre
	rm -rf draw
	rm -rf GAM
	rm -rf GBM
	rm -rf __pycache__

.SECONDARY:
.DELETE_ON_ERROR:

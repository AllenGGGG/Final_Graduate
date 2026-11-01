"""Publish the current point-1 manuscript only after evidence verification."""
from pathlib import Path
import json
from slice_alloc.tools.modal_binding_run import OUT,ROOT,read,write,sha
from slice_alloc.tools.export_document import to_docx


def main():
    verification=read(OUT/'verification.json');assert verification['passed']
    design=(ROOT/'slice_alloc/design.md').read_text()
    review=(OUT/'design_review.md').read_text()
    formulation=(OUT/'formulation.md').read_text()
    report=(OUT/'report.md').read_text()
    contents=['# 面向低空智联的多模态网络切片技术','',
        '研究项点：（1）低空智联下的多模态网络切片智能分配；（2）低空智联下的多模态网络切片资源预测；（3）低空智联下的多模态网络切片QoS保障。本版完成研究点1的设计、原生实现、训练及留出验证，后二者尚未实施。','',
        '## 研究背景与问题','',
        '低空无人机的移动接入、共享任务内容和异构SLA使统一承载方式面临移动连续性、内容复用和资源开销之间的取舍。多模态网络强调在共享基础设施中共存不同网络制式；网络切片强调业务合同、功能部署、资源预约和生命周期管理。本文将业务到模态绑定与切片初始准入/部署联合决定，而非把多模态等同于视频、图像等数据类型。','',
        '本版不是三章算法均已完成的全文终稿；它是研究点1可交付的最终设计与实验正文。','',
        '低空场景依据[王云涛等，低空智联网架构、安全与优化关键技术，电子与信息学报，2026，48(3):889–913](https://doi.org/10.11999/JEIT250947)中的空地网络与资源协同问题；12机双AP等参数是本课题可复现的受限建模选择，并非该综述直接给出的真实部署配置。','',
        design.replace('# 面向低空智联的多模态网络切片技术：研究点1最终设计','## 系统设计与实验合同',1), '',
        review.replace('# 旧提案复核结论','## 设计合理性复核',1),'',
        formulation.replace('# 可写入论文的研究点1数学表述','## 决策模型与求解方法',1),'',
        report.replace('# 面向低空智联的多模态网络切片技术：研究点1最终实验报告','## 原生仿真实验结果',1),'',
        '## 可复现材料','',
        '[当前设计](slice_alloc/design.md) · [正式报告](slice_alloc/results/modal_binding/report.md) · [完整统计](slice_alloc/results/modal_binding/analysis.json) · [证据验真](slice_alloc/results/modal_binding/verification.json) · [复现操作](slice_alloc/results/modal_binding/REPRODUCE.md)','',
        '结果以冻结协议和逐包证据为准，不宣称全局最优或跨所有场景优越。第二、第三研究点只规定后续输入输出，不能引用未发生的实验成功率。']
    text='\n'.join(contents)+'\n'
    text=text.replace('(figures/','(slice_alloc/results/modal_binding/figures/')
    # Report-local evidence paths need to become repository-relative in manuscript.
    for name in ('protocol.json','selection.json','analysis.json','evaluation.json'):
        text=text.replace(f']({name})',f'](slice_alloc/results/modal_binding/{name})')
    text=text.replace('](../../design.md)','](slice_alloc/design.md)')
    md=ROOT/'面向低空智联的多模态网络切片技术.md';md.write_text(text)
    docx=ROOT/'面向低空智联的多模态网络切片技术.docx';to_docx(text,destination=docx)
    readme=ROOT/'README.md'
    readme.write_text(readme.read_text()+'\n[研究点1完整正文（Markdown）](面向低空智联的多模态网络切片技术.md) · [Word正文](面向低空智联的多模态网络切片技术.docx)\n')
    provenance={str(path.relative_to(ROOT)):sha(path) for path in [md,docx,OUT/'report.md',OUT/'report.docx',
        ROOT/'slice_alloc/tools/modal_binding_report.py',ROOT/'slice_alloc/tools/modal_binding_verify.py',
        ROOT/'slice_alloc/tools/modal_binding_publish.py',ROOT/'slice_alloc/tools/modal_binding_challengers.py',
        ROOT/'slice_alloc/tools/modal_binding_allocate.py',OUT/'recommended_policy.json',
        ROOT/'slice_alloc/modal_binding_contract.schema.json']}
    write(OUT/'delivery.json',dict(completed_research_points=[1],unimplemented_research_points=[2,3],
        native_evidence_verified=True,artifacts=provenance,training_interactions=491520,
        audited_evaluation_episodes=verification['audited_episodes']))
    print('PUBLISHED',md,docx)

if __name__=='__main__':main()

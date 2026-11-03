// Device-level deficit round robin: native IP and NDN use the same serializer.
// Included inside the executor's anonymous namespace, after Task and Message.
uint32_t ClassifySlice(Ptr<const Packet> packet) {
 auto p=packet->Copy();PppHeader ppp;p->RemoveHeader(ppp);
 if(ppp.GetProtocol()==0x21){
  Ipv4Header ip;p->RemoveHeader(ip);if(ip.GetProtocol()!=17)return 0;
  UdpHeader udp;p->RemoveHeader(udp);if(udp.GetDestinationPort()!=4341||p->GetSize()<9)return 0;
  Message h;p->RemoveHeader(h);return tasks.count(h.sid)?h.sid:0;
 }
 try {
  ns3::ndn::BlockHeader h;p->RemoveHeader(h);auto block=h.getBlock();
  if(block.type()!=5&&block.type()!=6){block.parse();for(const auto& e:block.elements())if(e.type()==80){block=e.blockFromValue();break;}}
  std::string name;
  if(block.type()==5)name=::ndn::Interest(block).getName().toUri();
  else if(block.type()==6)name=::ndn::Data(block).getName().toUri();
  // A multicast/cache-shared physical packet is charged once to its earliest
  // active requesting slice. It is never duplicated merely for accounting.
  for(auto& x:tasks)if(x.second.accepted&&x.second.mode=="ndn"&&Now()<x.second.drain&&name.find(x.second.prefix+"/")==0)return x.first;
 }catch(const std::exception&){}
 return 0;
}
class SliceQueue:public Queue<Packet>{
public:
 static TypeId GetTypeId(){static TypeId t=TypeId("ns3::SliceQueue<Packet>").SetParent<Queue<Packet>>().AddConstructor<SliceQueue>()
  .AddAttribute("MaxSize","Shared buffer capacity",QueueSizeValue(QueueSize("100p")),MakeQueueSizeAccessor(&QueueBase::SetMaxSize,&QueueBase::GetMaxSize),MakeQueueSizeChecker());return t;}
 uint32_t node=0,peer=0;double capacity=1;uint64_t input=0,served=0,dropped=0,unclassified=0,changes=0;double waitSum=0;
 std::map<uint32_t,uint32_t> weight;std::map<uint32_t,double> changed;
 std::map<uint64_t,std::pair<uint32_t,double>> meta;
 std::map<uint32_t,int64_t> deficit;std::deque<uint32_t> active;bool credit=true;
 bool Enqueue(Ptr<Packet> p)override{
  const auto sid=ClassifySlice(p);input+=p->GetSize();if(!sid)++unclassified;
  Log(node,"queue_arrive",sid,peer,p->GetSize(),capacity);
  if(!DoEnqueue(end(),p)){dropped+=p->GetSize();Log(node,"queue_drop",sid,peer,p->GetSize());return false;}
  meta[p->GetUid()]={sid,Now()};if(std::find(active.begin(),active.end(),sid)==active.end()){active.push_back(sid);deficit[sid]=0;}
  return true;
 }
 Ptr<Packet> Dequeue()override{
  if(IsEmpty())return nullptr;
  for(uint32_t tries=0;tries<10000;++tries){
   NS_ABORT_MSG_IF(active.empty(),"DRR lost active class");const auto sid=active.front();
   auto it=begin();for(;it!=end();++it)if(meta.at((*it)->GetUid()).first==sid)break;
   if(it==end()){active.pop_front();deficit.erase(sid);credit=true;continue;}
   if(credit){deficit[sid]+=512*std::max(1u,weight[sid]);credit=false;}
   if(deficit[sid]<(*it)->GetSize()){active.pop_front();active.push_back(sid);credit=true;continue;}
   auto p=*it;deficit[sid]-=p->GetSize();auto waited=Now()-meta.at(p->GetUid()).second;meta.erase(p->GetUid());
   served+=p->GetSize();waitSum+=waited;Log(node,"queue_depart",sid,peer,p->GetSize(),waited);
   return DoDequeue(it);
  }
  NS_FATAL_ERROR("DRR did not make progress");return nullptr;
 }
 Ptr<Packet> Remove()override{if(IsEmpty())return nullptr;auto p=*begin();meta.erase(p->GetUid());return DoRemove(begin());}
 Ptr<const Packet> Peek()const override{return IsEmpty()?nullptr:DoPeek(begin());}
 std::map<uint32_t,double> HeadWait()const{
  std::map<uint32_t,double> heads;for(auto it=begin();it!=end();++it){auto m=meta.at((*it)->GetUid());heads[m.first]=std::max(heads[m.first],Now()-m.second);}return heads;
 }
 void Update(const std::string& mode,double forecast){
  auto heads=HeadWait();std::set<uint32_t> ids;for(auto& x:weight)ids.insert(x.first);for(auto& x:heads)ids.insert(x.first);
  for(auto sid:ids){uint32_t next=1;auto task=tasks.find(sid);
   if(mode!="none"&&task!=tasks.end()){
    const double deadline=task->second.deadline,wait=heads.count(sid)?heads[sid]:0.;
    if(wait>.25*deadline||(mode=="predictive"&&forecast>.8&&wait>.1*deadline))next=4;
   }
   uint32_t old=std::max(1u,weight[sid]);if(old!=next&&Now()-changed[sid]>=.5){weight[sid]=next;changed[sid]=Now();++changes;Log(node,"queue_weight",sid,peer,next,forecast);}
  }
 }
};
NS_OBJECT_ENSURE_REGISTERED(SliceQueue);
